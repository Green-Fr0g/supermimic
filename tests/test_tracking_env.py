"""Real Newton contracts; provide SUPERMIMIC_TEST_MOTION_DIR with canonical NPZs."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import parse_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.tasks.super_motion_tracking.mdp import rewards, terminations

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


@pytest.fixture(scope="module")
def tracking_env(tmp_path_factory):
    source = os.environ.get("SUPERMIMIC_TEST_MOTION_DIR")
    if not source:
        pytest.skip("Set SUPERMIMIC_TEST_MOTION_DIR to run real Newton integration")
    files = sorted(Path(source).rglob("*.npz"))
    if not files:
        pytest.fail(f"No NPZ files in {source}")
    # Short real clips make end boundaries deterministic and quick to exercise.
    motion_dir = tmp_path_factory.mktemp("tracking_motions")
    with np.load(files[0], allow_pickle=False) as data:
        frame_count = len(data["dof_positions"])
        for clip, length in enumerate((6, 54)):
            assert frame_count >= length
            arrays = {
                key: data[key][:length]
                if key
                in (
                    "dof_positions",
                    "dof_velocities",
                    "body_positions",
                    "body_rotations",
                    "body_linear_velocities",
                    "body_angular_velocities",
                )
                else data[key]
                for key in data.files
            }
            np.savez(motion_dir / f"clip{clip}.npz", **arrays)
    wp.config.enable_backward = False
    cfg = parse_env_cfg("SuperMimic-MotionTracking-G1", device="cuda:0", num_envs=2)
    cfg.seed = 42
    cfg.commands.motion.motion_dir = str(motion_dir)
    cfg.sim.physics.solver_cfg.save_to_mjcf = str(motion_dir / "model.xml")
    with launch_simulation(cfg, {"headless": True, "visualizer": []}):
        env = gym.make("SuperMimic-MotionTracking-G1", cfg=cfg).unwrapped
        yield env
        env.close()


def test_reference_reset_rewards_context_and_terminal_boundary(tracking_env):
    """Exercise real reset/physics/auto-reset and the collector-facing terminal state."""
    env = tracking_env
    cmd = env.command_manager.get_term("motion")
    obs, _ = env.reset(seed=42)
    assert {key: tuple(value.shape) for key, value in obs.items()} == {
        "policy": (2, 930),
        "goal": (2, 710),
        "goal_mask": (2, 10),
        "state": (2, 71),
        "context": (2, 33),
    }
    history = obs["policy"].reshape(2, 10, 93)
    torch.testing.assert_close(history, history[:, :1].expand_as(history))
    torch.testing.assert_close(cmd.robot.data.joint_pos.torch[:, cmd.joint_ids], cmd.joint_pos)
    torch.testing.assert_close(
        cmd.robot_body_pos_w[:, cmd.root_index], cmd.body_pos_w[:, cmd.root_index], atol=1e-5, rtol=0
    )
    for term in (rewards.root_position, rewards.root_orientation, rewards.body_position, rewards.body_orientation):
        torch.testing.assert_close(term(env, std=0.3), torch.ones(2, device=env.device), atol=2e-4, rtol=0)

    # Actual episode parameters must change only for reset environments.
    original_context = cmd.context.clone()
    original_com = cmd.robot.data.body_com_pos_b.torch.clone()
    torso_cfg = env.event_manager.get_term_cfg("torso_com")
    torso_ids = torso_cfg.params["asset_cfg"].body_ids
    nominal_com = original_com - original_context[:, None, -3:]
    env.reset(env_ids=torch.tensor([0], device=env.device))
    assert not torch.equal(cmd.context[0], original_context[0])
    torch.testing.assert_close(cmd.context[1], original_context[1])
    torch.testing.assert_close(cmd.robot.data.body_com_pos_b.torch[1], original_com[1])
    for _ in range(3):
        env.reset(env_ids=torch.tensor([0], device=env.device))
        assert 0.3 <= cmd.context[0, 0] <= 1.2
        assert cmd.context[0, 1:30].abs().max() <= 0.01
        offset = cmd.robot.data.body_com_pos_b.torch[:, torso_ids] - nominal_com[:, torso_ids]
        torch.testing.assert_close(offset, cmd.context[:, None, -3:], atol=1e-6, rtol=0)

    # Set a known final transition. The reset writes frame zero of a new episode;
    # the post-step command update then advances its reference to frame one.
    cmd.cfg.evaluation = True
    cmd.cfg.evaluation_motion_id = 0
    for name in ("physics_material", "joint_calibration", "torso_com"):
        env.event_manager.get_term_cfg(name).params["enabled"] = False
    env.reset()
    cmd.cfg.evaluation_motion_id = 1
    env.reset(env_ids=torch.tensor([1], device=env.device))
    assert cmd.motion_ids.tolist() == [0, 1]
    cmd.cfg.evaluation_motion_id = 0
    cmd.time_steps[0] = 5
    old_context = cmd.context.clone()
    ongoing_history = env.obs_buf["policy"][1].reshape(10, 93).clone()
    obs, reward, failed, truncated, extras = env.step(torch.zeros(2, 29, device=env.device))
    assert truncated.tolist() == [True, False]
    assert extras["transition"]["motion_end"].tolist() == [True, False]
    assert extras["transition"]["next_frame"].tolist() == [5, 1]
    assert cmd.time_steps.tolist() == [1, 1]
    assert not extras["final_obs"]["goal_mask"].any()
    assert obs["goal_mask"].sum() == 14  # four frames remain after the reset update, ten in the longer clip
    torch.testing.assert_close(extras["transition"]["context"], old_context)
    assert extras["final_obs"]["policy"].shape == (1, 930)
    assert extras["final_env_ids"].tolist() == [0]
    assert all(torch.isfinite(value).all() for value in obs.values())
    assert torch.isfinite(reward).all()
    history = obs["policy"].reshape(2, 10, 93)
    torch.testing.assert_close(history[0], history[0, :1].expand_as(history[0]))
    torch.testing.assert_close(history[1, :-1], ongoing_history[1:])

    # Failure is pelvis-only: perturb its height/tilt and check physical flags,
    # independently of the motion-end truncation above.
    pose = torch.cat((cmd.body_pos_w[:, cmd.root_index], cmd.body_quat_w[:, cmd.root_index]), -1)
    pose[0, 2] += 0.3
    pose[1, 3:] = pose.new_tensor([1.0, 0.0, 0.0, 0.0])  # upside-down xyzw
    cmd.robot.write_root_pose_to_sim_index(root_pose=pose)
    env.sim.forward()
    assert terminations.root_height_error(env, 0.25).tolist() == [True, False]
    assert terminations.root_tilt_error(env, 0.8)[1]

    # A perfect one-step clip/time truncation has the same sampler return as a
    # long perfect rollout. A physical failure gets no tail bootstrap.
    cmd.cfg.evaluation = False
    cmd.sampler.update_rate = 1.0
    cmd.start_bins[:] = torch.arange(2, device=env.device)
    cmd.episode_returns.zero_()
    cmd.episode_reward_sum.zero_()
    cmd.episode_steps.zero_()
    cmd.discount.fill_(1)
    env.reset_terminated[:] = torch.tensor([False, True], device=env.device)
    cmd.record_rewards(torch.full((2,), 5 * env.step_dt, device=env.device), torch.arange(2, device=env.device))
    cmd.finish_returns(torch.arange(2, device=env.device))
    torch.testing.assert_close(
        cmd.sampler.returns[:2],
        torch.tensor([5 * env.step_dt / (1 - cmd.cfg.discount), 5 * env.step_dt], device=env.device),
    )


def test_runtime_com_dynamics_matches_fresh_model(tracking_env, tmp_path):
    """Runtime CoM must refresh inertia/bias, not just the public pose buffer."""
    env = tracking_env
    cmd = env.command_manager.get_term("motion")
    event_cfg = env.event_manager.get_term_cfg("torso_com")
    for name in ("physics_material", "joint_calibration", "torso_com"):
        env.event_manager.get_term_cfg(name).params["enabled"] = True
    cmd.cfg.evaluation = True
    env.reset(seed=73)
    nominal_com = cmd.robot.data.body_com_pos_b.torch.clone() - cmd.context[:, None, -3:]
    env.step(torch.zeros(2, 29, device=env.device))  # processes the model-change notification
    solver = env.sim.physics_manager._solver
    native_model, native_data = solver.mjw_model, solver.mjw_data
    # Recompute at the exact qpos/qvel exported below, without integration.
    with wp.ScopedDevice(env.device):
        mjw.fwd_position(native_model, native_data)
        mjw.fwd_velocity(native_model, native_data)
        mjw.fwd_actuation(native_model, native_data)
        mjw.fwd_acceleration(native_model, native_data)
    torso = next(i for i in range(solver.mj_model.nbody) if "torso_link" in solver.mj_model.body(i).name)
    torso_name = solver.mj_model.body(torso).name
    torso_id = event_cfg.params["asset_cfg"].body_ids[0]
    expected_com = (nominal_com[0, torso_id] + cmd.context[0, -3:]).cpu().numpy()

    # Recompile from MJCF with the same independently specified CoM, rather than
    # copying refreshed constants from the runtime model under test.
    xml_path = tmp_path / "fresh.xml"
    tree = ET.parse(env.cfg.sim.physics.solver_cfg.save_to_mjcf)
    body = next(body for body in tree.iter("body") if body.get("name") == torso_name)
    body.find("inertial").set("pos", " ".join(str(float(v)) for v in expected_com))
    tree.write(xml_path)
    fresh = mujoco.MjModel.from_xml_path(str(xml_path))
    fresh_data = mujoco.MjData(fresh)
    fresh_data.qpos[:] = native_data.qpos.numpy()[0]
    fresh_data.qvel[:] = native_data.qvel.numpy()[0]
    fresh_data.ctrl[:] = native_data.ctrl.numpy()[0]
    fresh_data.qfrc_applied[:] = native_data.qfrc_applied.numpy()[0]
    fresh_data.xfrc_applied[:] = native_data.xfrc_applied.numpy()[0]
    mujoco.mj_forward(fresh, fresh_data)
    np.testing.assert_allclose(native_data.M.numpy()[0], fresh_data.M, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(native_data.qfrc_bias.numpy()[0], fresh_data.qfrc_bias, atol=5e-4, rtol=2e-4)
    np.testing.assert_allclose(native_data.qacc_smooth.numpy()[0], fresh_data.qacc_smooth, atol=1e-3, rtol=5e-4)
    # These constants feed contact impedance and are easily left stale by a
    # partial runtime update even when body_ipos itself looks correct.
    np.testing.assert_allclose(native_model.body_subtreemass.numpy()[0], fresh.body_subtreemass, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(native_model.body_invweight0.numpy()[0], fresh.body_invweight0, atol=3e-5, rtol=2e-4)


def test_contact_friction_uses_episode_coefficient(tracking_env):
    """Ground contact must use sampled mu even when it is below ground mu=1."""
    env = tracking_env
    cmd = env.command_manager.get_term("motion")
    for name in ("physics_material", "joint_calibration", "torso_com"):
        env.event_manager.get_term_cfg(name).params["enabled"] = True
    env.reset(seed=42)
    pose = torch.cat((cmd.body_pos_w[:, cmd.root_index], cmd.body_quat_w[:, cmd.root_index]), -1)
    pose[:, 2] -= 0.08
    cmd.robot.write_root_pose_to_sim_index(root_pose=pose)
    env.step(torch.zeros(2, 29, device=env.device))
    solver = env.sim.physics_manager._solver
    count = solver.mjw_data.nacon.numpy()[0]
    contacts = solver.mjw_data.contact
    geoms = contacts.geom.numpy()[:count]
    world_ids = contacts.worldid.numpy()[:count]
    body_ids = solver.mj_model.geom_bodyid[geoms]
    ground_contacts = np.any(body_ids == 0, axis=-1)
    assert ground_contacts.any(), "Expected ground contact after lowering the feet"
    actual = contacts.friction.numpy()[:count][ground_contacts, 0]
    expected = cmd.context[:, 0].cpu().numpy()[world_ids[ground_contacts]]
    assert np.any(expected < 1), "This check must cover coefficients below ground mu"
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


def test_differentiable_kinematics_and_analytic_heads_match_simulator(tracking_env):
    """Independent simulator parity guards link/CoM frames and policy gradients."""
    from supermimic.tasks.super_motion_tracking.mdp.observations import proprioception
    from supermimic_rl.task_model import TrackingModel

    env = tracking_env
    cmd = env.command_manager.get_term("motion")
    cmd.cfg.evaluation = True
    cmd.cfg.evaluation_motion_id = 1
    for name in ("physics_material", "joint_calibration", "torso_com"):
        env.event_manager.get_term_cfg(name).params["enabled"] = True
    obs, _ = env.reset(seed=19)
    model = TrackingModel(env)
    previous = torch.zeros(2, 29, device=env.device)
    for _ in range(3):
        action = torch.randn_like(previous) * 0.1
        previous_state = obs["state"].clone()
        obs, reward, terminated, truncated, extras = env.step(action)
        assert not (terminated | truncated).any()
        state = obs["state"].detach().requires_grad_(True)
        p, q, v, w = model.fk(state, obs["context"])
        torch.testing.assert_close(p, cmd.robot_body_pos_w - env.scene.env_origins[:, None], atol=3e-5, rtol=1e-4)
        # Quaternion signs are physically interchangeable.
        assert (q * cmd.robot_body_quat_w).sum(-1).abs().min() > 0.99999
        torch.testing.assert_close(v, cmd.robot.data.body_lin_vel_w.torch[:, cmd.body_ids], atol=3e-4, rtol=2e-4)
        torch.testing.assert_close(w, cmd.robot.data.body_ang_vel_w.torch[:, cmd.body_ids], atol=3e-4, rtol=2e-4)
        predicted = model.reward(
            state, action, previous, obs["context"], cmd.motion_ids, extras["transition"]["frame"], previous_state
        )
        torch.testing.assert_close(predicted, reward, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(model.proprioception(state, action, obs["context"]), proprioception(env))
        torch.testing.assert_close(
            model.termination(state, cmd.motion_ids, extras["transition"]["frame"], soft=False), terminated
        )
        goal, mask = model.goal(cmd.motion_ids, cmd.time_steps)
        torch.testing.assert_close(goal, obs["goal"])
        torch.testing.assert_close(mask, obs["goal_mask"].bool())
        (gradient,) = torch.autograd.grad(predicted.sum(), state)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        previous = action


def test_world_fit_bptt_freezing_and_checkpoint_resume(tracking_env, tmp_path):
    """Real replay can be fit; frozen model Jacobians train actor; resume continues."""
    from supermimic_rl.config import LearningCfg
    from supermimic_rl.learner import frozen
    from supermimic_rl.runner import SuperMimicRunner

    env = tracking_env
    cmd = env.command_manager.get_term("motion")
    cmd.cfg.evaluation = True
    cmd.cfg.evaluation_motion_id = 1
    for name in ("physics_material", "joint_calibration", "torso_com"):
        env.event_manager.get_term_cfg(name).params["enabled"] = False
    cfg = LearningCfg.smoke()
    cfg.num_envs = 2
    cfg.prefill = 20
    cfg.replay_steps = 64
    cfg.dynamics_hidden = (64, 64)
    cfg.actor_hidden = (64, 32)
    cfg.encoder_hidden = (64, 32)
    cfg.critic_hidden = (32,)
    cfg.model_batches = 1
    cfg.learning_rate = 0.001
    runner = SuperMimicRunner(env, cfg, tmp_path)
    for _ in range(cfg.prefill):
        runner.collect()
    losses = []
    for _ in range(30):
        torch.manual_seed(912)  # Fit the same real windows, not a moving evaluation target.
        losses.append(runner.learner.fit_world(runner.replay)["world_loss"])
    assert losses[-1] < losses[0] * 0.8
    learner = runner.learner
    batch = {key: value.to(env.device) for key, value in runner.replay.sample(2).items()}
    state, context = batch["state"][:, 0], batch["context"][:, 0]
    # Isolate the dynamics Jacobian from the direct action-smoothing reward.
    action = torch.zeros(2, 29, device=env.device, requires_grad=True)
    with frozen(learner.world, learner.critic):
        next_state = learner.world(state, action, context)
        torch.testing.assert_close(next_state[:, 3:7].norm(dim=-1), torch.ones(2, device=env.device))
        (gradient,) = torch.autograd.grad(next_state[:, :3].sum(), action)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        world_before = {k: v.clone() for k, v in learner.world.state_dict().items()}
        critic_before = {k: v.clone() for k, v in learner.critic.state_dict().items()}
        loss, _ = learner.imagine(batch)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.actor.parameters())
        assert all(p.grad is None for p in learner.world.parameters())
        assert all(p.grad is None for p in learner.critic.parameters())
        for key, value in learner.world.state_dict().items():
            torch.testing.assert_close(value, world_before[key], rtol=0, atol=0)
        for key, value in learner.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[key], rtol=0, atol=0)
    runner.learner.improve_policy(runner.replay)
    runner.learner.advance_schedule()
    runner.iteration = 1
    path = tmp_path / "resume.pt"
    runner.save(path)
    obs = {key: value.clone() for key, value in runner.obs.items()}
    expected = runner.learner.act(obs).detach().clone()
    for p in runner.learner.parameters():
        p.data.zero_()
    runner.load(str(path))
    torch.testing.assert_close(runner.learner.act(obs), expected, atol=0, rtol=0)
    assert runner.iteration == 1 and runner.learner.schedulers["world"].last_epoch == 1
    runner.collect()
    metrics = runner.learner.fit_world(runner.replay)
    assert torch.isfinite(torch.tensor(metrics["world_loss"]))
    before = {
        name: [p.detach().clone() for p in module.parameters()]
        for name, module in (("world", learner.world), ("actor", learner.actor), ("critic", learner.critic))
    }
    runner.train(2)
    assert runner.iteration == 2
    for name, module in (("world", learner.world), ("actor", learner.actor), ("critic", learner.critic)):
        assert any(not torch.equal(old, p) for old, p in zip(before[name], module.parameters(), strict=True))
