"""NPZ mapping and clip isolation, independent of a simulator and external data."""

import numpy as np
import pytest
import torch

from supermimic.tasks.super_motion_tracking.mdp.commands import MotionLoader, ReturnBasedSampler


@pytest.fixture
def motion_dir(tmp_path):
    """Two distinguishable 50 Hz clips with deliberately non-task name order."""
    for clip, length in enumerate((53, 7)):
        joints = np.tile([12.0 + clip, 24.0 + clip], (length, 1)).astype(np.float32)
        position = np.zeros((length, 2, 3), dtype=np.float32)
        position[:, 1, 0] = np.arange(length) + clip * 100
        quaternion = np.zeros((length, 2, 4), dtype=np.float32)
        quaternion[..., 3] = 1
        np.savez(
            tmp_path / f"clip{clip}.npz",
            fps=np.array([50]),
            robot_name=np.array(["test"]),
            root_link_name=np.array(["root"]),
            dof_names=np.array(["b", "a"]),
            body_names=np.array(["tip", "root"]),
            dof_positions=joints,
            dof_velocities=joints * 2,
            body_positions=position,
            body_rotations=quaternion,
            body_linear_velocities=position * 3,
            body_angular_velocities=position * 4,
        )
    return tmp_path


@pytest.mark.unit
def test_library_name_mapping_and_clip_end_padding(motion_dir):
    """Changing clip length/order must not mix joints or leak into the next clip."""
    library = MotionLoader(motion_dir, ["a", "b"], ["root"], "root", "test")
    ids = library.indices(torch.tensor([[0], [1]]), torch.tensor([[51, 52, 53, 70], [0, 5, 6, 8]]))
    torch.testing.assert_close(
        library.body_positions[ids, 0, 0], torch.tensor([[51.0, 52.0, 52.0, 52.0], [100.0, 105.0, 106.0, 106.0]])
    )
    torch.testing.assert_close(library.dof_positions[ids[:, 0]], torch.tensor([[24.0, 12.0], [25.0, 13.0]]))
    torch.testing.assert_close(library.bin_motion, torch.tensor([0, 0, 1]))
    torch.testing.assert_close(library.bin_length, torch.tensor([50, 2, 6]))


@pytest.mark.unit
def test_library_rejects_wrong_names_and_rate(motion_dir):
    with pytest.raises(ValueError, match="missing names"):
        MotionLoader(motion_dir, ["absent"], ["root"], "root", "test")
    with pytest.raises(ValueError, match="expected 30"):
        MotionLoader(motion_dir, ["a"], ["root"], "root", "test", fps=30)


@pytest.mark.unit
def test_reset_distribution_keeps_uniform_floor_and_remembers_difficulty():
    """Low-return bins gain probability while every motion remains sampleable."""
    sampler = ReturnBasedSampler(4, "cpu", max_return=10, update_rate=1)
    torch.testing.assert_close(sampler.probabilities, torch.full((4,), 0.25))
    sampler.update(torch.tensor([0, 0, 1, 2, 3]), torch.tensor([0.0, 0.0, 10.0, 10.0, 10.0]))
    assert sampler.probabilities[0] > 0.5
    assert torch.all(sampler.probabilities >= 0.125)
    sampler.update(torch.tensor([0]), torch.tensor([10.0]))
    assert sampler.probabilities[0] > sampler.probabilities[1]
    assert sampler.probabilities.sum().item() == pytest.approx(1)
