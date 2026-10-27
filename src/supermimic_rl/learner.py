"""Paper return equations and a small learner-facing API."""

from contextlib import contextmanager

import torch


def survival_objective(
    rewards: torch.Tensor, survival: torch.Tensor, terminal_value: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Compute the differentiable soft-survival return used by SuperMimic."""
    result = torch.zeros_like(terminal_value)
    continuation = torch.ones_like(terminal_value)
    for step in range(rewards.shape[-1]):
        result = result + continuation * (gamma**step) * rewards[..., step]
        continuation = continuation * (1.0 - survival[..., step])
    result = result + continuation * (gamma ** rewards.shape[-1]) * terminal_value
    return result


def lambda_returns(
    rewards: torch.Tensor, survival: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute detached lambda targets with soft survival probabilities."""
    targets = torch.zeros_like(rewards)
    next_target = values[..., -1].detach()
    for step in reversed(range(rewards.shape[-1])):
        bootstrap = (1.0 - lam) * values[..., step + 1] + lam * next_target
        targets[..., step] = rewards[..., step] + gamma * (1.0 - survival[..., step]) * bootstrap
        next_target = targets[..., step]
    return targets.detach()


@contextmanager
def frozen(*modules):
    """Temporarily disable parameter gradients while preserving input gradients."""
    requires_grad = [[parameter.requires_grad for parameter in module.parameters()] for module in modules]
    try:
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        yield
    finally:
        for module, flags in zip(modules, requires_grad, strict=True):
            for parameter, flag in zip(module.parameters(), flags, strict=True):
                parameter.requires_grad_(flag)


class SuperMimicLearner(torch.nn.Module):
    """Joint autoregressive dynamics fitting and actor/critic imagination updates."""

    def __init__(self, task, cfg) -> None:
        super().__init__()
        import math

        from .networks import Critic, DeterministicActor, WorldModel

        self.task, self.cfg = task, cfg
        self.device = torch.device(cfg.device)
        joints = task.joints
        goal_dim = task.future_steps * (2 * joints + 13) + task.future_steps
        self.actor = DeterministicActor(3 * joints + 6, goal_dim, joints, cfg)
        self.world = WorldModel(2 * joints + 13, joints, joints + 4, cfg)
        # Heading-invariant state, reference root error, relative orientation, goal, xi and a_prev.
        self.critic = Critic((2 * joints + 11) + 3 + 4 + goal_dim + joints + 4 + joints, cfg)
        self.to(self.device)
        self.optimizers = {
            "world": torch.optim.Adam(self.world.parameters(), lr=cfg.learning_rate),
            "actor": torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95)),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate),
        }

        def decay(step):
            return cfg.final_lr_fraction + (1 - cfg.final_lr_fraction) * 0.5 * (
                1 + math.cos(math.pi * min(step / cfg.decay_iterations, 1))
            )

        self.schedulers = {
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, decay) for name, optimizer in self.optimizers.items()
        }

    def _goal(self, motion_ids, frames):
        goal, mask = self.task.goal(motion_ids, frames)
        return torch.cat((goal, mask.float()), -1)

    def critic_input(self, state, context, motion_ids, frames, previous_action):
        from .math import heading_quat, quat_apply, quat_inverse, quat_mul_xyzw, state_features

        ids = self.task.motion.indices(motion_ids, frames)
        ref_pos = self.task.motion.body_positions[ids, self.task.root]
        ref_quat = self.task.motion.body_rotations[ids, self.task.root]
        quat = state[..., 3:7]
        root_error = quat_apply(quat_inverse(heading_quat(quat)), ref_pos - state[..., :3])
        relative = quat_mul_xyzw(quat_inverse(quat), ref_quat)
        relative = torch.where(relative[..., 3:] < 0, -relative, relative)
        return torch.cat(
            (state_features(state), root_error, relative, self._goal(motion_ids, frames), context, previous_action), -1
        )

    def act(self, obs):
        history = obs["policy"].reshape(-1, self.task.history_length, 3 * self.task.joints + 6)
        goal = torch.cat((obs["goal"], obs["goal_mask"]), -1)
        return self.actor(history, goal)

    @torch.no_grad()
    def observe(self, obs, action, next_state, transition):
        """Update normalizers from executed real transitions only, before optimization."""
        from .math import state_difference

        state, context = obs["state"], transition["context"]
        history = obs["policy"].reshape(-1, self.task.history_length, 3 * self.task.joints + 6)
        self.actor.obs_norm.update(history[:, -1])
        self.actor.goal_norm.update(torch.cat((obs["goal"], obs["goal_mask"]), -1))
        self.world.in_norm.update(self.world.inputs(state, action, context))
        self.world.out_norm.update(state_difference(state, next_state))
        self.critic.norm.update(
            self.critic_input(
                state, context, transition["motion_id"], transition["frame"], history[:, -1, -self.task.joints :]
            )
        )

    def _batch(self, replay, count, starts=False):
        values = replay.sample_starts(count) if starts else replay.sample(count)
        return {key: value.to(self.device) for key, value in values.items()}

    def _step(self, name, parameters, clip_value=None):
        """Reject non-finite updates before they can poison model parameters."""
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.cfg.max_grad_norm)
        if not torch.isfinite(norm):
            self.optimizers[name].zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite {name} gradients")
        if clip_value is not None:
            torch.nn.utils.clip_grad_value_(parameters, clip_value)
        self.optimizers[name].step()
        self.optimizers[name].zero_grad(set_to_none=True)
        return float(norm)

    def fit_world(self, replay):
        """Equation 1: k-step autoregressive state loss; task heads are analytic."""
        from .math import state_difference

        metrics = []
        self.world.train()
        for _ in range(self.cfg.model_batches):
            batch = self._batch(replay, self.cfg.model_batch)
            optimizer = self.optimizers["world"]
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for start in range(0, self.cfg.model_batch, self.cfg.microbatch):
                part = {key: value[start : start + self.cfg.microbatch] for key, value in batch.items()}
                state = part["state"][:, 0]
                loss = torch.zeros((), device=self.device)
                for step in range(self.cfg.window):
                    prediction = self.world(state, part["action"][:, step], part["context"][:, step])
                    error = state_difference(part["next_state"][:, step], prediction) / self.world.out_norm.std
                    mask = part["mask"][:, step]
                    loss = loss + (error.square().mean(-1) * mask).sum() / self.cfg.model_batch
                    state = torch.where(mask[:, None], prediction, state)
                loss.backward()
                loss_total += float(loss.detach())
            self._step("world", list(self.world.parameters()))
            metrics.append(loss_total / self.cfg.window)
        return {"world_loss": sum(metrics) / len(metrics)}

    def imagine(self, batch):
        """Equation 2, retaining dynamics and critic input Jacobians through all H steps.

        Call with frozen(world, critic) during actor optimization. Targets for the
        subsequent critic fit are detached from this graph.
        """
        state, context = batch["state"][:, 0], batch["context"][:, 0]
        frames, motion_ids = batch["frame"][:, 0], batch["motion_id"][:, 0]
        previous_action = batch["previous_action"][:, 0]
        history = batch["history"]
        inputs = [self.critic_input(state, context, motion_ids, frames, previous_action)]
        rewards, terminations = [], []
        lengths = self.task.motion.lengths[motion_ids]
        for _ in range(self.cfg.horizon):
            action = self.actor(history, self._goal(motion_ids, frames))
            next_state = self.world(state, action, context)
            reward = self.task.reward(next_state, action, previous_action, context, motion_ids, frames, state)
            terminal = self.task.termination(next_state, motion_ids, frames)
            valid = frames < lengths
            reward = torch.where(valid, reward, 0)
            # Clip ends are known boundaries, not learned physical-failure labels.
            terminal = torch.where(frames + 1 >= lengths, 1, terminal)
            next_obs = self.task.proprioception(next_state, action, context, noisy=True)
            history = torch.cat((history[:, 1:], next_obs[:, None]), 1)
            state, previous_action, frames = next_state, action, frames + 1
            inputs.append(self.critic_input(state, context, motion_ids, frames, previous_action))
            rewards.append(reward)
            terminations.append(terminal)
        rewards = torch.stack(rewards, -1)
        terminations = torch.stack(terminations, -1)
        inputs = torch.stack(inputs, 1)
        terminal_value = self.critic(inputs[:, -1])
        objective = survival_objective(rewards, terminations, terminal_value, self.cfg.gamma)
        with torch.no_grad():
            values = self.critic(inputs.detach())
            targets = lambda_returns(rewards, terminations, values, self.cfg.gamma, self.cfg.gae_lambda)
            valid_frames = batch["frame"][:, :1] + torch.arange(self.cfg.horizon, device=self.device) < lengths[:, None]
        return -objective.mean(), {
            "inputs": inputs[:, :-1].detach(),
            "targets": targets,
            "weights": valid_frames.float(),
            "imagined_reward": rewards.detach().mean(),
        }

    def improve_policy(self, replay):
        """One actor update, then fixed lambda targets over shuffled critic minibatches."""
        batch = self._batch(replay, self.cfg.imagination_batch, starts=True)
        self.optimizers["actor"].zero_grad(set_to_none=True)
        self.optimizers["world"].zero_grad(set_to_none=True)
        self.optimizers["critic"].zero_grad(set_to_none=True)
        rollouts = []
        actor_loss = 0.0
        with frozen(self.world, self.critic):
            for start in range(0, self.cfg.imagination_batch, self.cfg.microbatch):
                part = {key: value[start : start + self.cfg.microbatch] for key, value in batch.items()}
                loss, rollout = self.imagine(part)
                weight = len(part["state"]) / self.cfg.imagination_batch
                (loss * weight).backward()
                actor_loss += float(loss.detach()) * weight
                rollouts.append(rollout)
        actor_norm = self._step("actor", list(self.actor.parameters()), self.cfg.actor_gradient_value)
        inputs = torch.cat([r["inputs"] for r in rollouts]).flatten(0, 1)
        targets = torch.cat([r["targets"] for r in rollouts]).flatten()
        weights = torch.cat([r["weights"] for r in rollouts]).flatten()
        critic_loss = 0.0
        updates = 0
        for _ in range(self.cfg.critic_epochs):
            indices = torch.randperm(len(inputs), device=self.device)
            for selected in indices.tensor_split(min(self.cfg.critic_minibatches, len(inputs))):
                self.optimizers["critic"].zero_grad(set_to_none=True)
                # Equation 3 averages trajectories and sums valid imagined steps.
                # Soft survival weights belong to the actor objective only.
                denominator = len(selected) / self.cfg.horizon
                value_loss = 0.0
                for chunk in selected.split(self.cfg.microbatch):
                    loss = ((self.critic(inputs[chunk]) - targets[chunk]).square() * weights[chunk]).sum() / denominator
                    loss.backward()
                    value_loss += float(loss.detach())
                self._step("critic", list(self.critic.parameters()))
                critic_loss += value_loss
                updates += 1
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss / updates,
            "actor_grad_norm": actor_norm,
            "imagined_reward": float(torch.stack([r["imagined_reward"] for r in rollouts]).mean()),
        }

    def advance_schedule(self):
        for scheduler in self.schedulers.values():
            scheduler.step()
