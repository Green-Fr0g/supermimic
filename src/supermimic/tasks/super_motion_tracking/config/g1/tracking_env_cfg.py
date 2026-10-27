"""G1 asset, joint ordering, body set, and low-gain actuation specialization."""

from isaaclab.utils import configclass
from isaaclab_newton.sim.schemas import MujocoCollisionCfg

from supermimic.assets.robots.g1 import (
    G1_ANCHOR_LINK_NAME,
    G1_DOF_ORDER,
    G1_ROOT_LINK_NAME,
    MIMIC_G1_ACTION_SCALE,
    MIMIC_G1_CFG,
)

from ...tracking_env_cfg import TrackingEnvCfg


@configclass
class G1TrackingEnvCfg(TrackingEnvCfg):
    """29-DOF G1, retaining BeyondMimic's fourteen tracked bodies."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = MIMIC_G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Native MuJoCo otherwise takes max(robot_mu, ground_mu), clipping
        # the intended [0.3, 1.2] distribution below the ground's mu=1.
        self.scene.robot.spawn.collision_props = [MujocoCollisionCfg(priority=1)]
        self.scene.robot.spawn.make_uninstanceable = True
        # Restore the BeyondMimic nominal pose and motor inertias. z_lab's
        # general-purpose G1 pose/constant armature differ from that baseline.
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.2,
        }
        self.scene.robot.soft_joint_pos_limit_factor = 0.9
        self.scene.robot.actuators["mimic_motors"].armature = {
            ".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint": 0.010177520,
            ".*_hip_roll_joint|.*_knee_joint": 0.025101925,
            ".*_ankle_.*|waist_pitch_joint|waist_roll_joint": 2 * 0.003609725,
            ".*_shoulder_.*|.*_elbow_joint|.*_wrist_roll_joint": 0.003609725,
            ".*_wrist_pitch_joint|.*_wrist_yaw_joint": 0.00425,
        }
        self.actions.joint_pos.joint_names = list(G1_DOF_ORDER)
        self.actions.joint_pos.scale = MIMIC_G1_ACTION_SCALE
        self.commands.motion.robot_name = "g1"
        self.commands.motion.joint_names = list(G1_DOF_ORDER)
        self.commands.motion.root_body_name = G1_ROOT_LINK_NAME
        self.commands.motion.anchor_body_name = G1_ANCHOR_LINK_NAME
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        self.events.torso_com.params["asset_cfg"].body_names = ["torso_link"]
