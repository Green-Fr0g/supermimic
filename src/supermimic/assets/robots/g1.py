"""Configuration for the Unitree G1 29-DOF humanoid (Isaac Lab 3.0)."""

import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from supermimic.assets import SUPERMIMIC_ASSETS_DIR

G1_USD_PATH = SUPERMIMIC_ASSETS_DIR / "models" / "g1" / "g1_29dof_rev_1_0.usd"

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(G1_USD_PATH),
        fix_root_link=False,
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            joint_effort_limit=88.0,
            actuator_velocity_limit=32.0,
            stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0},
            damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0},
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            joint_effort_limit=139.0,
            actuator_velocity_limit=20.0,
            stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
            damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            joint_effort_limit=25.0,
            actuator_velocity_limit=37.0,
            stiffness=40.0,
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow_.*": 1.0,
                ".*_wrist_roll.*": 1.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            joint_effort_limit=5.0,
            actuator_velocity_limit=22.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)

G1_DOF_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_AMP_DOF_ORDER = G1_DOF_ORDER

G1_ROOT_LINK_NAME = "pelvis"
G1_ANCHOR_LINK_NAME = "torso_link"
G1_AMP_LINK_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def _g1_actuator_value(joint_name: str, attribute: str) -> float | None:
    """Look up a per-joint actuator value the way IsaacLab resolves it.

    The actuator group whose ``joint_names_expr`` matches the joint owns it, and within that group
    the first regex pattern that fully matches the joint wins. Entries matching no pattern resolve
    to 0.0, which leaves the joint undriven (i.e. free-floating).
    """
    for actuator_cfg in G1_CFG.actuators.values():
        if not any(re.fullmatch(pattern, joint_name) for pattern in actuator_cfg.joint_names_expr):
            continue
        values = getattr(actuator_cfg, attribute, None)
        if values is None:
            return None
        if not isinstance(values, dict):
            return float(values)
        for pattern, value in values.items():
            if re.fullmatch(pattern, joint_name):
                return float(value)
        return 0.0
    return None


def _g1_action_scale() -> dict[str, float]:
    """Map each controlled joint to its action scale, ``0.25 * effort_limit / stiffness``."""
    scales = {}
    for joint_name in list(G1_DOF_ORDER):
        effort_limit = _g1_actuator_value(joint_name, "joint_effort_limit")
        stiffness = _g1_actuator_value(joint_name, "stiffness")
        if effort_limit and stiffness:
            scales[joint_name] = 0.25 * effort_limit / stiffness
    return scales


G1_ACTION_SCALE = _g1_action_scale()


MIMIC_G1_CFG = G1_CFG.replace(
    actuators={
        "mimic_motors": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            joint_effort_limit={
                ".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint": 88.0,
                ".*_hip_roll_joint|.*_knee_joint": 139.0,
                ".*_ankle_.*|waist_[rp]itch_joint|waist_roll_joint": 50.0,
                ".*_shoulder_.*|.*_elbow_joint|.*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint|.*_wrist_yaw_joint": 5.0,
            },
            actuator_velocity_limit={
                ".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint": 32.0,
                ".*_hip_roll_joint|.*_knee_joint": 20.0,
                ".*_ankle_.*|waist_[rp]itch_joint|waist_roll_joint": 37.0,
                ".*_shoulder_.*|.*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint|.*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint": 40.2,
                ".*_hip_roll_joint|.*_knee_joint": 99.1,
                ".*_ankle_.*|waist_[rp]itch_joint|waist_roll_joint": 28.5,
                ".*_shoulder_.*|.*_elbow_joint|.*_wrist_roll_joint": 14.3,
                ".*_wrist_pitch_joint|.*_wrist_yaw_joint": 16.8,
            },
            damping={
                ".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint": 2.6,
                ".*_hip_roll_joint|.*_knee_joint": 6.3,
                ".*_ankle_.*|waist_[rp]itch_joint|waist_roll_joint": 1.8,
                ".*_shoulder_.*|.*_elbow_joint|.*_wrist_roll_joint": 0.9,
                ".*_wrist_pitch_joint|.*_wrist_yaw_joint": 1.1,
            },
            armature=0.01,
        ),
    }
)


def _resolve_actuator_value(cfg, joint_name: str, attribute: str) -> float | None:
    values = getattr(cfg, attribute, None)
    if values is None:
        return None
    if not isinstance(values, dict):
        return float(values)
    for pattern, value in values.items():
        if re.fullmatch(pattern, joint_name):
            return float(value)
    return None


def _mimic_g1_action_scale() -> dict[str, float]:
    scales = {}
    cfg = MIMIC_G1_CFG.actuators["mimic_motors"]
    for joint_name in G1_DOF_ORDER:
        effort = _resolve_actuator_value(cfg, joint_name, "joint_effort_limit")
        stiffness = _resolve_actuator_value(cfg, joint_name, "stiffness")
        if effort and stiffness:
            scales[joint_name] = 0.25 * effort / stiffness
    return scales


MIMIC_G1_ACTION_SCALE = _mimic_g1_action_scale()
