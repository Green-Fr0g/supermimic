"""Unitree G1 settings for the standalone SuperMimic trainer."""

from isaaclab.utils import configclass

from ....rl_cfg import (
    SuperMimicActorCfg,
    SuperMimicAlgorithmCfg,
    SuperMimicRunnerCfg,
    SuperMimicWorldModelCfg,
)


@configclass
class G1SuperMimicRunnerCfg(SuperMimicRunnerCfg):
    """Paper networks and batches; environment count belongs to the task config."""

    # Iterations count joint updates; UTD controls vectorized simulator steps per update.
    max_iterations = 320_000
    save_interval = 2500
    experiment_name = "g1_supermimic"
    seed = 42
    actor = SuperMimicActorCfg(
        hidden_dims=[2048, 1024, 512, 256],
        encoder_hidden_dims=[2048, 1024, 512, 256],
        activation="elu",
        obs_normalization=True,
        latent_size=32,
    )
    critic_hidden_dims = [256, 128]
    world_model = SuperMimicWorldModelCfg(
        hidden_dims=[2048, 2048, 2048],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = SuperMimicAlgorithmCfg()


@configclass
class G1SuperMimicSmokeRunnerCfg(G1SuperMimicRunnerCfg):
    """Reduced batches for configuration checks, preserving the paper networks.

    Use 16 environments via the environment/CLI configuration. Selecting an
    agent configuration does not change the number of simulator environments.
    """

    max_iterations = 25
    save_interval = 25
    experiment_name = "g1_supermimic_smoke"
    algorithm = SuperMimicAlgorithmCfg(model_batch=8, imagination_batch=8, microbatch=8)
