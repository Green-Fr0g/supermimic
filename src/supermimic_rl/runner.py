"""Standalone collection, joint updates, checkpoints and deterministic evaluation."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import LearningCfg
from .learner import SuperMimicLearner
from .networks import DeterministicActor
from .replay import ReplayBuffer
from .task_model import TrackingModel


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SuperMimicRunner:
    """Own simulation collection and the paper's collect/fit/improve loop.

    Inference mode creates only the actor; it neither allocates nor loads replay.
    """

    def __init__(self, env, cfg: LearningCfg, log_dir: str | Path, inference: bool = False):
        cfg.validate()
        if env.num_envs != cfg.num_envs:
            raise ValueError("Runner and simulator environment counts differ")
        if torch.device(env.device) != torch.device(cfg.device):
            raise ValueError("Runner and simulator must use the same device")
        self.env, self.cfg, self.inference = env, cfg, inference
        self.device = torch.device(cfg.device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        observation, _ = env.reset(seed=cfg.seed)
        self.obs = {key: value.clone() for key, value in observation.items()}
        self.task = TrackingModel(env)
        if self.task.history_length != 10:
            raise ValueError("This policy implements the paper's ten-frame history")
        goal_dim = self.task.future_steps * (2 * self.task.joints + 13) + self.task.future_steps
        self.learner = None if inference else SuperMimicLearner(self.task, cfg)
        self.actor = (
            DeterministicActor(3 * self.task.joints + 6, goal_dim, self.task.joints, cfg).to(self.device)
            if inference
            else self.learner.actor
        )
        self.replay = None if inference else ReplayBuffer(cfg.num_envs, cfg.replay_steps, cfg.window)
        self.iteration, self.sim_steps, self.step_credit = 0, 0, 0.0
        cmd = env.command_manager.get_term("motion")
        physics = env.cfg.sim.physics.to_dict()
        if "solver_cfg" in physics:
            physics["solver_cfg"].pop("save_to_mjcf", None)
        robot_cfg = env.cfg.scene.robot.to_dict()
        robot_cfg["spawn"].pop("usd_path", None)
        self.contract = json.loads(
            json.dumps(
                {
                    "joint_names": cmd.cfg.joint_names,
                    "body_names": cmd.cfg.body_names,
                    "anchor": cmd.cfg.anchor_body_name,
                    "root": cmd.cfg.root_body_name,
                    "step_dt": env.step_dt,
                    "physics": physics,
                    "robot": robot_cfg,
                    "asset_layers": sorted(_file_hash(path) for path in self.task.asset_layers),
                    "quaternion": "xyzw",
                    "motions": [
                        {"name": name, "length": int(length), "sha256": _file_hash(path)}
                        for name, length, path in zip(
                            cmd.motion.names, cmd.motion.lengths.cpu(), cmd.motion.files, strict=True
                        )
                    ],
                },
                default=str,
            )
        )
        if not inference:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
            (self.log_dir / "environment.json").write_text(json.dumps(self.contract, indent=2))

    @torch.no_grad()
    def collect(self):
        if self.inference:
            raise RuntimeError("Inference runner cannot collect replay")
        action = self.learner.act(self.obs)
        executed = action + self.cfg.action_noise * torch.randn_like(action)
        if not torch.isfinite(executed).all():
            raise FloatingPointError("Non-finite simulator actions")
        next_obs, reward, terminated, truncated, extras = self.env.step(executed)
        self.replay.add(self.obs, executed, next_obs, reward, terminated, truncated, extras)
        next_state = next_obs["state"].clone()
        if "final_env_ids" in extras:
            next_state[extras["final_env_ids"].long()] = extras["final_obs"]["state"]
        self.learner.observe(self.obs, executed, next_state, extras["transition"])
        self.obs = {key: value.clone() for key, value in next_obs.items()}
        self.sim_steps += self.env.num_envs
        return {
            "real_reward": float(reward.mean()),
            "failure_fraction": float(terminated.float().mean()),
            "boundary_fraction": float((terminated | truncated).float().mean()),
        }

    def train(self, iterations: int) -> Path:
        """Train to an absolute total number of joint updates, including after resume."""
        if self.inference or iterations < self.iteration:
            raise ValueError("Invalid training mode or total iteration target")
        if self.replay.size == 0:
            for _ in range(self.cfg.prefill):
                self.collect()
        parameters = sum(p.numel() for p in self.learner.parameters())
        budget = {
            "parameters": parameters,
            "parameter_bytes": parameters * 4,
            "adam_moments_estimated_bytes": parameters * 8,
            "replay_capacity_bytes": sum(data.numel() * data.element_size() for data in self.replay._data.values()),
        }
        (self.log_dir / "memory_budget.json").write_text(json.dumps(budget, indent=2))
        with (self.log_dir / "metrics.jsonl").open("a") as metrics_file:
            while self.iteration < iterations:
                # Every update collects 1/UTD vectorized steps, with fractional carry.
                self.step_credit += 1 / self.cfg.update_to_data
                steps = int(self.step_credit)
                self.step_credit -= steps
                started = time.perf_counter()
                collection = [self.collect() for _ in range(steps)]
                collected = time.perf_counter()
                metrics = {key: sum(row[key] for row in collection) / steps for key in collection[0]}
                metrics.update(self.learner.fit_world(self.replay))
                fitted = time.perf_counter()
                metrics.update(self.learner.improve_policy(self.replay))
                improved = time.perf_counter()
                self.learner.advance_schedule()
                self.iteration += 1
                metrics.update(
                    iteration=self.iteration,
                    sim_steps=self.sim_steps,
                    learning_rate=self.learner.optimizers["actor"].param_groups[0]["lr"],
                    torch_peak_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
                    collection_seconds=collected - started,
                    model_seconds=fitted - collected,
                    policy_critic_seconds=improved - fitted,
                )
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"Non-finite training metrics: {metrics}")
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                print(
                    f"update {self.iteration}: world={metrics['world_loss']:.5f} "
                    f"actor={metrics['actor_loss']:.5f} critic={metrics['critic_loss']:.5f}",
                    flush=True,
                )
                if self.iteration % self.cfg.save_interval == 0:
                    self.save(self.log_dir / f"checkpoint_{self.iteration}.pt")
        path = self.log_dir / "checkpoint_final.pt"
        self.save(path)
        return path

    @torch.no_grad()
    def play(self, steps: int):
        """Run deterministic actions without constructing or modifying replay."""
        self.actor.eval()
        for _ in range(steps):
            history = self.obs["policy"].reshape(-1, 10, 3 * self.task.joints + 6)
            action = self.actor(history, torch.cat((self.obs["goal"], self.obs["goal_mask"]), -1))
            if not torch.isfinite(action).all():
                raise FloatingPointError("Non-finite playback actions")
            self.obs, reward, *_ = self.env.step(action)
            if not torch.isfinite(self.obs["state"]).all() or not torch.isfinite(reward).all():
                raise FloatingPointError("Non-finite playback state or reward")

    @torch.no_grad()
    def evaluate(self, max_steps: int | None = None):
        """Play every NPZ once from its first frame; keep failures separate from truncations."""
        if self.env.num_envs != 1 or not self.env.command_manager.get_term("motion").cfg.evaluation:
            raise ValueError("Full evaluation requires one environment in play mode")
        cmd = self.env.command_manager.get_term("motion")
        results = []
        for motion_id, name in enumerate(cmd.motion.names):
            cmd.cfg.evaluation_motion_id = motion_id
            self.obs, _ = self.env.reset()
            total, failed, partial, count = 0.0, False, False, 0
            root_error, body_error = 0.0, 0.0
            # BeyondMimic timing executes one reward-bearing transition per reference index.
            budget = int(cmd.motion.lengths[motion_id])
            for _ in range(budget):
                history = self.obs["policy"].reshape(1, 10, -1)
                action = self.actor(history, torch.cat((self.obs["goal"], self.obs["goal_mask"]), -1))
                previous_frame = cmd.time_steps.clone()
                previous_state, context = self.obs["state"].clone(), self.obs["context"].clone()
                self.obs, reward, terminated, truncated, extras = self.env.step(action)
                total += float(reward[0])
                count += 1
                state = extras["final_obs"]["state"] if bool((terminated | truncated)[0]) else self.obs["state"]
                ids = self.task.motion.indices(torch.tensor([motion_id], device=self.device), previous_frame)
                root_error += float((state[0, :3] - self.task.motion.body_positions[ids, self.task.root][0]).norm())
                motion_ids = torch.tensor([motion_id], device=self.device)
                targets, _ = self.task.aligned_targets(previous_state, context, motion_ids, previous_frame)
                positions, *_ = self.task.fk(state, context)
                body_error += float((targets - positions).norm(dim=-1).mean())
                # Count root error on the actual executed part, including the pre-reset final state.
                if bool((terminated | truncated)[0]):
                    failed = bool(terminated[0])
                    break
                if max_steps is not None and count >= max_steps:
                    partial = True
                    break
            results.append(
                {
                    "motion": name,
                    "steps": count,
                    "failed": failed,
                    "partial": partial,
                    "normalized_return": total / (5 * self.task.dt * budget),
                    "root_position_error": root_error / count,
                    "body_position_error": body_error / count,
                }
            )
        return {
            "motions": results,
            "failure_count": sum(row["failed"] for row in results),
            "normalized_return": sum(row["normalized_return"] for row in results) / len(results),
            "partial": any(row["partial"] for row in results),
        }

    def save(self, path):
        if self.inference:
            raise RuntimeError("Cannot save a training checkpoint from an inference runner")
        sampler = self.env.command_manager.get_term("motion").sampler
        (self.log_dir / "sampler.json").write_text(
            json.dumps(
                {
                    "returns": sampler.returns.tolist(),
                    "visits": sampler.visits.tolist(),
                    "probabilities": sampler.probabilities.tolist(),
                }
            )
        )
        payload = {
            "version": 1,
            "config": asdict(self.cfg),
            "contract": self.contract,
            "actor": self.actor.state_dict(),
            "learner": self.learner.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.learner.optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in self.learner.schedulers.items()},
            "replay": self.replay.state_dict(),
            "iteration": self.iteration,
            "sim_steps": self.sim_steps,
            "step_credit": self.step_credit,
            "sampler": {k: getattr(sampler, k).cpu() for k in ("returns", "history", "visits")},
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        policy_path = path.with_suffix(".policy.pt")
        policy_temporary = policy_path.with_suffix(policy_path.suffix + ".tmp")
        torch.save({key: payload[key] for key in ("version", "config", "contract", "actor")}, policy_temporary)
        policy_temporary.replace(policy_path)

    def load(self, path):
        """Restore on CPU, then move networks only; inference ignores all training state."""
        path = Path(path)
        if self.inference and not path.stem.endswith(".policy"):
            path = path.with_suffix(".policy.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("version") != 1:
            raise ValueError("Unsupported checkpoint; collection-only checkpoints cannot resume training")
        if state["contract"] != self.contract:
            raise ValueError("Checkpoint task, robot, physics or motion dataset differs")
        self.actor.load_state_dict(state["actor"])
        if self.inference:
            self.actor.eval()
            return
        current, saved = asdict(self.cfg), state["config"]
        for key in current:
            if key not in ("iterations", "save_interval", "device", "microbatch") and current[key] != saved[key]:
                raise ValueError(f"Checkpoint training configuration differs: {key}")
        self.learner.load_state_dict(state["learner"])
        for name in self.learner.optimizers:
            self.learner.optimizers[name].load_state_dict(state["optimizers"][name])
            self.learner.schedulers[name].load_state_dict(state["schedulers"][name])
        self.replay.load_state_dict(state["replay"])
        self.replay.new_collection()
        self.iteration, self.sim_steps, self.step_credit = state["iteration"], state["sim_steps"], state["step_credit"]
        sampler = self.env.command_manager.get_term("motion").sampler
        for name, value in state["sampler"].items():
            getattr(sampler, name).copy_(value)
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        observation, _ = self.env.reset(seed=None)
        self.obs = {key: value.clone() for key, value in observation.items()}
