"""Motion encoder/decoder, geometric dynamics, and critic (paper Table A.4)."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .math import integrate_state, state_features


def mlp(input_dim: int, output_dim: int, hidden_dims, activation: str = "elu") -> nn.Sequential:
    activation_cls = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), activation_cls()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Explicit real-data updates; optimization never changes running statistics."""

    def __init__(self, dim: int, enabled: bool = True, floor: float = 0.01):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("floor", torch.full((dim,), floor))

    @torch.no_grad()
    def update(self, value):
        if not self.enabled or self.count >= 2e8:
            return
        value = value.reshape(-1, value.shape[-1])
        mean, var = value.mean(0), value.var(0, unbiased=False)
        total = self.count + len(value)
        delta = mean - self.mean
        self.var.copy_(
            (self.var * self.count + var * len(value) + delta.square() * self.count * len(value) / total) / total
        )
        self.mean.add_(delta * len(value) / total)
        self.count.copy_(total)

    @property
    def std(self):
        return self.var.sqrt().clamp_min(self.floor) if self.enabled else torch.ones_like(self.mean)

    def forward(self, value):
        return (value - self.mean) / self.std if self.enabled else value


def _forward(net, value):
    # Non-reentrant checkpoint retains input/parameter gradients without storing all MLP activations.
    return checkpoint(net, value, use_reentrant=False) if torch.is_grad_enabled() else net(value)


class DeterministicActor(nn.Module):
    """Encode reference into bounded z; decode history and z into unbounded normalized actions."""

    def __init__(self, proprio_dim, goal_dim, action_dim, cfg):
        super().__init__()
        self.obs_norm = RunningNorm(proprio_dim, cfg.obs_normalization)
        self.goal_norm = RunningNorm(goal_dim, cfg.obs_normalization)
        self.encoder = mlp(goal_dim, cfg.latent_size, cfg.encoder_hidden, cfg.activation)
        self.decoder = mlp(proprio_dim * 10 + cfg.latent_size, action_dim, cfg.actor_hidden, cfg.activation)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.01)
            self.decoder[-1].bias.zero_()

    def forward(self, history, goal):
        z = torch.tanh(_forward(self.encoder, self.goal_norm(goal)))
        inputs = torch.cat((self.obs_norm(history).flatten(-2), z), -1)
        return _forward(self.decoder, inputs)


class WorldModel(nn.Module):
    """Heading-invariant dynamics with local residuals and SO(3) integration."""

    def __init__(self, state_dim, action_dim, context_dim, cfg):
        super().__init__()
        self.in_norm = RunningNorm(state_dim - 2 + action_dim + context_dim, cfg.world_normalization)
        self.out_norm = RunningNorm(state_dim - 1, cfg.world_normalization, floor=1e-4)
        self.net = mlp(
            state_dim - 2 + action_dim + context_dim, state_dim - 1, cfg.dynamics_hidden, cfg.world_activation
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
            self.out_norm.floor.copy_(
                torch.tensor(
                    [0.003] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3 + [0.003] * action_dim + [0.05] * action_dim
                )
            )

    def inputs(self, state, action, context):
        return torch.cat((state_features(state), action, context), -1)

    def forward(self, state, action, context):
        delta = _forward(self.net, self.in_norm(self.inputs(state, action, context)))
        delta = delta * self.out_norm.std + self.out_norm.mean if self.out_norm.enabled else delta
        return integrate_state(state, delta)


class Critic(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.norm = RunningNorm(input_dim, cfg.obs_normalization)
        self.net = mlp(input_dim, 1, cfg.critic_hidden, cfg.activation)

    def forward(self, inputs):
        return _forward(self.net, self.norm(inputs)).squeeze(-1)
