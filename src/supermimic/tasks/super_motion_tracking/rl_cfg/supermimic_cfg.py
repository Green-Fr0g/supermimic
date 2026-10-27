"""Configuration schemas for the standalone :mod:`supermimic_rl` trainer."""

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class SuperMimicModelCfg:
    """Common MLP model settings consumed by ``supermimic_rl``."""

    hidden_dims: list[int] = MISSING
    activation: str = "elu"
    obs_normalization: bool = True


@configclass
class SuperMimicActorCfg(SuperMimicModelCfg):
    """Deterministic history-conditioned actor settings."""

    encoder_hidden_dims: list[int] = MISSING
    latent_size: int = 32


@configclass
class SuperMimicWorldModelCfg(SuperMimicModelCfg):
    """Geometric dynamics; reward, observation and soft termination heads are analytic."""


@configclass
class SuperMimicAlgorithmCfg:
    """Paper training parameters consumed by the standalone learner."""

    prefill: int = 128
    replay_steps: int = 1000
    action_noise: float = 0.1
    update_to_data: float = 0.4
    window: int = 8
    model_batch: int = 2048
    model_batches: int = 10
    horizon: int = 32
    imagination_batch: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    learning_rate: float = 2.0e-4
    decay_iterations: int = 250_000
    final_lr_fraction: float = 0.03
    max_grad_norm: float = 1.0
    actor_gradient_value: float = 0.1
    microbatch: int = 256


@configclass
class SuperMimicRunnerCfg:
    """Standalone train/play configuration, independent of any RL framework."""

    device: str = "cuda:0"
    seed: int = 42
    max_iterations: int = 320_000
    save_interval: int = 2500
    experiment_name: str = "g1_supermimic"
    actor: SuperMimicActorCfg = MISSING
    critic_hidden_dims: list[int] = MISSING
    world_model: SuperMimicWorldModelCfg = MISSING
    algorithm: SuperMimicAlgorithmCfg = MISSING
