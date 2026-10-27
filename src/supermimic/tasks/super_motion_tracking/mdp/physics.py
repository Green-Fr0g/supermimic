"""Newton MJWarp preset at 200 Hz, with no hidden extra physics substeps."""

from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from isaaclab_tasks.utils import PresetCfg


@configclass
class TrackingSolverCfg(MJWarpSolverCfg):
    """Use dense Jacobians for the small G1 system.

    MJWarp auto selects sparse at nv>32. Its deterministic sparse Hessian
    reduction allocates a multi-GB temporary already at 16 G1 environments.
    Dense avoids this workspace without reducing contacts or solver accuracy.
    """

    jacobian: str = "dense"


@configclass
class TrackingPhysicsCfg(PresetCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=TrackingSolverCfg(
            njmax=300,
            nconmax=80,
            integrator="implicitfast",
            cone="pyramidal",
            update_data_interval=1,
            use_mujoco_contacts=False,
            disable_sensors=True,
        ),
        num_substeps=1,
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=1_000_000),
        deterministic_mode="run_to_run",
    )
    default: NewtonCfg = newton_mjwarp
