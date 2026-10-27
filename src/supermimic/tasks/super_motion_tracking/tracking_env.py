"""Manager-based environment with reference timing and collection boundaries."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .mdp.observations import proprioception, proprioception_noise


class MotionTrackingEnv(ManagerBasedRLEnv):
    """Expose tracking transitions and terminal observations around the base step.

    The managers still own rewards, events, actions and observations. This small
    lifecycle adapter follows BeyondMimic's command timing: the current reference
    frame is used for physics, reward and termination, and the command term advances
    it after the base environment has completed that work.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._tracking_step = False
        super().__init__(*args, **kwargs)

    def step(self, action: torch.Tensor) -> tuple:
        cmd = self.command_manager.get_term("motion")
        self.extras.pop("final_obs", None)
        self.extras.pop("final_env_ids", None)
        transition = {
            "motion_id": cmd.motion_ids.clone(),
            "frame": cmd.time_steps.clone(),
            "context": cmd.context.clone(),
        }
        next_frame = cmd.time_steps + 1
        next_frame = next_frame.minimum(cmd.motion.lengths[cmd.motion_ids] - 1)
        transition["next_frame"] = next_frame
        transition["motion_end"] = cmd.motion_ended.clone()
        self._tracking_step = True
        try:
            result = super().step(action)
        finally:
            self._tracking_step = False
        done = result[2] | result[3]
        cmd.record_rewards(result[1], (~done).nonzero().flatten())
        self.extras["transition"] = transition
        return result

    def _reset_idx(self, env_ids: torch.Tensor | slice) -> None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if self._tracking_step:
            cmd = self.command_manager.get_term("motion")
            cmd.record_rewards(self.reward_buf, env_ids)
            cmd.finish_returns(env_ids)
            # ObservationManager.compute(False) returns the stored history. Build
            # the terminal history from the previous history plus the new frame,
            # so ongoing environments' history is never appended twice.
            final = {key: value[env_ids].clone() for key, value in self.observation_manager.compute().items()}
            current = proprioception(self)
            if self.cfg.observations.policy.enable_corruption:
                current = proprioception_noise(current, self.cfg.observations.policy.proprioception.noise)
            history = self.obs_buf["policy"][env_ids].reshape(
                len(env_ids), self.cfg.observations.policy.history_length, -1
            )
            final["policy"] = torch.cat((history[:, 1:], current[env_ids, None]), dim=1).flatten(1)
            self.extras["final_env_ids"] = env_ids.clone()
            self.extras["final_obs"] = final
        super()._reset_idx(env_ids)
        # Core auto-reset does not forward kinematics before command/observation
        # computation. Refresh without taking an extra physics step.
        self.scene.write_data_to_sim()
        self.sim.forward()
