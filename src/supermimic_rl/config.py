"""Validated runtime configuration, resolved from the task's registered configuration."""

from dataclasses import dataclass


@dataclass
class LearningCfg:
    """Flatten the task configuration for simulator-independent algorithm code.

    Iterations count joint world-model / policy / critic updates, not simulator steps.
    """

    profile: str = "smoke"
    num_envs: int = 16
    iterations: int = 25
    prefill: int = 128
    replay_steps: int = 1000
    model_batch: int = 8
    model_batches: int = 10
    imagination_batch: int = 8
    window: int = 8
    horizon: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2e-4
    action_noise: float = 0.1
    update_to_data: float = 0.4
    seed: int = 42
    device: str = "cuda:0"
    save_interval: int = 1000
    dynamics_hidden: tuple[int, ...] = (2048, 2048, 2048)
    actor_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    latent_size: int = 32
    activation: str = "elu"
    obs_normalization: bool = True
    world_activation: str = "elu"
    world_normalization: bool = True
    critic_epochs: int = 5
    critic_minibatches: int = 4
    decay_iterations: int = 250_000
    final_lr_fraction: float = 0.03
    max_grad_norm: float = 1.0
    actor_gradient_value: float = 0.1
    microbatch: int = 256

    @classmethod
    def from_task(cls, task, num_envs: int, profile: str) -> "LearningCfg":
        cfg = cls(
            **vars(task.algorithm),
            num_envs=num_envs,
            profile=profile,
            iterations=task.max_iterations,
            seed=task.seed,
            device=task.device,
            save_interval=task.save_interval,
            actor_hidden=tuple(task.actor.hidden_dims),
            encoder_hidden=tuple(task.actor.encoder_hidden_dims),
            dynamics_hidden=tuple(task.world_model.hidden_dims),
            critic_hidden=tuple(task.critic_hidden_dims),
            latent_size=task.actor.latent_size,
            activation=task.actor.activation,
            obs_normalization=task.actor.obs_normalization,
            world_activation=task.world_model.activation,
            world_normalization=task.world_model.obs_normalization,
        )
        cfg.validate()
        return cfg

    @classmethod
    def smoke(cls) -> "LearningCfg":
        from supermimic.tasks.super_motion_tracking.config.g1.agents.supermimic_cfg import G1SuperMimicSmokeRunnerCfg

        return cls.from_task(G1SuperMimicSmokeRunnerCfg(), 16, "smoke")

    def validate(self) -> None:
        for name in (
            "num_envs",
            "replay_steps",
            "window",
            "horizon",
            "model_batch",
            "model_batches",
            "imagination_batch",
            "microbatch",
            "critic_epochs",
            "critic_minibatches",
            "save_interval",
            "prefill",
            "decay_iterations",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.replay_steps < self.window:
            raise ValueError("replay_steps must be at least window")
        if not 0 < self.update_to_data <= 1 or not 0 < self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid UTD, discount or lambda")
        if self.iterations < 0 or self.action_noise < 0 or self.learning_rate <= 0:
            raise ValueError("Invalid training length, noise or learning rate")
