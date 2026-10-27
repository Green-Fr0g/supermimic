"""Compact time-major replay with reset, overwrite, and resume boundaries."""

from __future__ import annotations

from typing import Any

import torch


class ReplayBuffer:
    """Store real transitions and reconstruct episode-local observation histories."""

    def __init__(self, num_envs: int, capacity: int, window: int, device: str = "cpu", history_length: int = 10):
        if num_envs < 1 or capacity < window or window < 1:
            raise ValueError("Replay requires num_envs > 0 and capacity >= window > 0")
        self.num_envs, self.capacity, self.window = num_envs, capacity, window
        self.history_length = history_length
        self.device = torch.device(device)
        self.write = 0
        self.size = 0
        self.episode = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_step = torch.zeros_like(self.episode)
        self._data: dict[str, torch.Tensor] = {}

    @property
    def total(self) -> int:
        return self.size * self.num_envs

    def add(self, obs, actions, next_obs, rewards, terminated, truncated, extras) -> None:
        """Capture executed actions and pre-reset next states; keep only one proprioceptive frame."""
        transition = extras["transition"]
        next_state = next_obs["state"].detach().to(self.device).clone()
        final_ids = extras.get("final_env_ids")
        if final_ids is not None and len(final_ids):
            next_state[final_ids.to(self.device).long()] = extras["final_obs"]["state"].detach().to(self.device)
        policy = obs["policy"].reshape(self.num_envs, self.history_length, -1)[:, -1]
        values = {
            "state": obs["state"],
            "next_state": next_state,
            "proprioception": policy,
            "previous_action": policy[:, -actions.shape[-1] :],
            "action": actions,
            "reward": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "motion_id": transition["motion_id"],
            "frame": transition["frame"],
            "context": transition["context"],
            "episode": self.episode,
            "episode_step": self.episode_step,
        }
        for name, value in values.items():
            value = value.detach().to(self.device)
            if name not in self._data:
                self._data[name] = torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=self.device)
            self._data[name][self.write].copy_(value)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        done = (terminated | truncated).to(self.device)
        self.episode += done.long()
        self.episode_step = torch.where(done, 0, self.episode_step + 1)

    def new_collection(self) -> None:
        """Keep old replay while starting a new physical episode after checkpoint restore."""
        self.episode += 1
        self.episode_step.zero_()

    def _sample(self, batch_size: int, length: int) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty replay")
        ages = torch.randint(self.size, (batch_size,), device=self.device)
        envs = torch.randint(self.num_envs, (batch_size,), device=self.device)
        offsets = torch.arange(length, device=self.device)
        safe_ages = (ages[:, None] + offsets).clamp_max(self.size - 1)
        rows = (self.write - self.size + safe_ages) % self.capacity
        out = {name: data[rows, envs[:, None]] for name, data in self._data.items()}
        valid = (ages[:, None] + offsets < self.size) & (out["episode"] == out["episode"][:, :1])
        out["mask"] = valid
        out["boundary"] = out["terminated"] | out["truncated"] | ~valid
        # Age limits prevent wraparound into the future when a long episode fills the ring.
        start_rows = rows[:, 0]
        back = torch.arange(self.history_length - 1, -1, -1, device=self.device)
        back = torch.minimum(back[None], self._data["episode_step"][start_rows, envs, None])
        back = torch.minimum(back, ages[:, None])
        history_rows = (start_rows[:, None] - back) % self.capacity
        out["history"] = self._data["proprioception"][history_rows, envs[:, None]]
        return out

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample autoregressive windows, masking transitions after reset or the newest stored step."""
        return self._sample(batch_size, self.window)

    def sample_starts(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample complete start-state records, including latest and pre-failure states."""
        return self._sample(batch_size, 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "capacity": self.capacity,
            "window": self.window,
            "history_length": self.history_length,
            "write": self.write,
            "size": self.size,
            "episode": self.episode.cpu(),
            "episode_step": self.episode_step.cpu(),
            # torch.save preserves a view's full storage. Clone a partially filled
            # ring so unused capacity is neither serialized nor read from memory.
            "data": {
                key: value[: self.size].cpu().clone() if self.size < self.capacity else value.cpu()
                for key, value in self._data.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("num_envs", "capacity", "window", "history_length"):
            if state[name] != getattr(self, name):
                raise ValueError(f"Replay {name} mismatch: {state[name]} != {getattr(self, name)}")
        self.write, self.size = state["write"], state["size"]
        self.episode = state["episode"].to(self.device).clone()
        self.episode_step = state["episode_step"].to(self.device).clone()
        self._data = {}
        for key, value in state["data"].items():
            storage = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.device)
            storage[: self.size].copy_(value.to(self.device))
            self._data[key] = storage
