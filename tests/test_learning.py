"""Replay episode boundaries and independent paper-equation/gradient contracts."""

import pytest
import torch

from supermimic_rl.learner import lambda_returns, survival_objective
from supermimic_rl.math import rotvec_quat
from supermimic_rl.replay import ReplayBuffer

pytestmark = pytest.mark.unit


def test_paper_survival_and_lambda_targets_preserve_input_gradients():
    rewards = torch.tensor([[2.0, 3.0]], requires_grad=True)
    dones = torch.tensor([[0.5, 0.2]], requires_grad=True)
    values = torch.tensor([[1.0, 2.0, 4.0]], requires_grad=True)
    objective = survival_objective(rewards, dones, values[:, -1], 0.9)
    torch.testing.assert_close(objective, torch.tensor([4.646]))
    targets = lambda_returns(rewards, dones, values, 0.9, 0.5)
    torch.testing.assert_close(targets, torch.tensor([[3.773, 5.88]]))
    assert not targets.requires_grad
    objective.sum().backward()
    torch.testing.assert_close(dones.grad, torch.tensor([[-5.292, -1.62]]))
    torch.testing.assert_close(values.grad, torch.tensor([[0.0, 0.0, 0.324]]))
    # Zero rotation increments must retain finite, non-zero orientation gradients.
    zero = torch.zeros(1, 3, requires_grad=True)
    rotvec_quat(zero)[:, :3].sum().backward()
    torch.testing.assert_close(zero.grad, torch.full_like(zero, 0.5))


def test_replay_terminal_capture_history_and_ring_overwrite():
    replay = ReplayBuffer(2, 24, 3)
    for t in range(50):
        value = torch.tensor([float(t), float(t + 1000)])
        state = value[:, None].expand(-1, 71)
        done = torch.tensor([t % 7 == 6, t % 9 == 8])
        final_ids = done.nonzero().flatten()
        returned = state + 1
        returned[final_ids] = -999
        replay.add(
            {"state": state, "policy": value[:, None].expand(-1, 930)},
            torch.zeros(2, 29),
            {"state": returned},
            torch.zeros(2),
            done,
            torch.zeros_like(done),
            {
                "transition": {
                    "motion_id": value.long() // 1000,
                    "frame": value.long(),
                    "context": value[:, None].expand(-1, 33),
                },
                "final_env_ids": final_ids,
                "final_obs": {"state": state[final_ids] + 1},
            },
        )
    torch.manual_seed(4)
    batch = replay.sample(512)
    starts = replay.sample_starts(512)
    assert int((starts["state"][:, 0, 0] % 1000).max()) == 49
    assert starts["terminated"].any()
    for samples in (batch, starts):
        torch.testing.assert_close(samples["state"], samples["state"][..., :1].expand_as(samples["state"]))
        torch.testing.assert_close(samples["state"][..., 0], samples["frame"].float())
        torch.testing.assert_close(samples["context"][..., 0], samples["frame"].float())
        valid = samples["mask"]
        torch.testing.assert_close(samples["next_state"][valid], samples["state"][valid] + 1)
        for states, history in zip(samples["state"], samples["history"], strict=True):
            env_id, time = divmod(int(states[0, 0]), 1000)
            episode_start = time // (7 if env_id == 0 else 9) * (7 if env_id == 0 else 9)
            expected = [max(episode_start, 26, past) + 1000 * env_id for past in range(time - 9, time + 1)]
            torch.testing.assert_close(history[:, 0], torch.tensor(expected, dtype=torch.float32))
    adjacent = batch["mask"][:, 1:]
    torch.testing.assert_close(
        (batch["state"][:, 1:, 0] - batch["state"][:, :-1, 0])[adjacent], torch.ones(int(adjacent.sum()))
    )
    long_episode = ReplayBuffer(1, 4, 3)
    for t in range(6):
        value = torch.full((1, 71), float(t))
        long_episode.add(
            {"state": value, "policy": value[:, :1].expand(-1, 930)},
            torch.zeros(1, 29),
            {"state": value + 1},
            torch.zeros(1),
            torch.tensor([False]),
            torch.tensor([False]),
            {
                "transition": {
                    "motion_id": torch.zeros(1, dtype=torch.long),
                    "frame": torch.tensor([t]),
                    "context": torch.zeros(1, 33),
                }
            },
        )
    sample = long_episode.sample_starts(128)
    for state, history in zip(sample["state"], sample["history"], strict=True):
        time = int(state[0, 0])
        torch.testing.assert_close(history[:, 0], torch.tensor([float(max(2, p)) for p in range(time - 9, time + 1)]))
    restored = ReplayBuffer(2, 24, 3)
    restored.load_state_dict(replay.state_dict())
    retained = restored.total
    restored.new_collection()
    assert restored.total == retained
    for t in range(2):
        state = torch.full((2, 71), float(5000 + t))
        restored.add(
            {"state": state, "policy": torch.full((2, 930), float(5000 + t))},
            torch.zeros(2, 29),
            {"state": state + 1},
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.tensor([t == 1, t == 1]),
            {
                "transition": {
                    "motion_id": torch.zeros(2, dtype=torch.long),
                    "frame": torch.full((2,), t),
                    "context": torch.zeros(2, 33),
                }
            },
        )
    samples = restored.sample(2048)
    assert (samples["state"][:, 0, 0] >= 5000).any()
    valid_adjacent = samples["mask"][:, 1:]
    differences = samples["state"][:, 1:, 0] - samples["state"][:, :-1, 0]
    torch.testing.assert_close(differences[valid_adjacent], torch.ones(int(valid_adjacent.sum())))
