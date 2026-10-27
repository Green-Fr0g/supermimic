"""Shared motion-tracking task; robot-specific settings live in config/<robot>.

Composition follows z_lab locomotion. Tracking terms derive from its migrated
BeyondMimic task, with the changes in SuperMimic Appendix A and D.1.
"""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from . import mdp

DEFAULT_MOTION_DIR = Path(__file__).resolve().parents[4] / "assets" / "motions"


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    """Flat ground and a robot provided by the concrete configuration."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(motion_dir=str(DEFAULT_MOTION_DIR))


@configclass
class ActionsCfg:
    joint_pos = mdp.CalibratedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Actor inputs stay separate from privileged collection/model inputs."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprioception = ObsTerm(func=mdp.proprioception, noise=NoiseCfg(func=mdp.proprioception_noise))

        def __post_init__(self) -> None:
            self.history_length = 10
            self.flatten_history_dim = True
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class GoalCfg(ObsGroup):
        reference = ObsTerm(func=mdp.future_reference)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class GoalMaskCfg(ObsGroup):
        valid = ObsTerm(func=mdp.future_reference_mask)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StateCfg(ObsGroup):
        state = ObsTerm(func=mdp.physical_state)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ContextCfg(ObsGroup):
        parameters = ObsTerm(func=mdp.sim_context)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    goal: GoalCfg = GoalCfg()
    goal_mask: GoalMaskCfg = GoalMaskCfg()
    state: StateCfg = StateCfg()
    context: ContextCfg = ContextCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_physics_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.3, 1.2),
            "restitution": 0.0,
            "enabled": True,
        },
    )
    joint_calibration = EventTerm(
        func=mdp.randomize_joint_calibration,
        mode="reset",
        params={"joint_range": (-0.01, 0.01), "enabled": True},
    )
    torso_com = EventTerm(
        func=mdp.randomize_torso_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": (0.025, 0.05, 0.05),
            "enabled": True,
        },
    )


@configclass
class RewardsCfg:
    body_position = RewTerm(func=mdp.body_position, weight=1.0, params={"std": 0.3})
    body_orientation = RewTerm(func=mdp.body_orientation, weight=1.0, params={"std": 0.4})
    body_linear_velocity = RewTerm(func=mdp.body_linear_velocity, weight=1.0, params={"std": 1.0})
    body_angular_velocity = RewTerm(func=mdp.body_angular_velocity, weight=1.0, params={"std": 3.14})
    root_position = RewTerm(func=mdp.root_position, weight=0.5, params={"std": 0.3})
    root_orientation = RewTerm(func=mdp.root_orientation, weight=0.5, params={"std": 0.4})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_end = DoneTerm(func=mdp.motion_end, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_error, params={"threshold": 0.25})
    root_tilt = DoneTerm(func=mdp.root_tilt_error, params={"threshold": 0.8})


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Robot-independent SuperMimic task at 50 Hz control / 200 Hz physics."""

    scene: TrackingSceneCfg = TrackingSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = mdp.TrackingPhysicsCfg()
        self.sim.physics_material = self.scene.terrain.physics_material
        # MotionTrackingEnv captures final state/history before resetting.
        self.compute_final_obs = False

    def play_mode(self) -> None:
        """Deterministic, noise-free full-clip playback from its first frame."""
        super().play_mode()
        self.events.physics_material.params["enabled"] = False
        self.events.joint_calibration.params["enabled"] = False
        self.events.torso_com.params["enabled"] = False
        self.commands.motion.evaluation = True
        self.terminations.time_out = None
