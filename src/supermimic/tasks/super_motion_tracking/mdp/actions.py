"""Position targets with episode-local calibration, without changing nominal q0."""

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class CalibratedJointPositionAction(JointPositionAction):
    """Apply q_sp = q0 + calibration + scale * a, in the configured joint order."""

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        command = self._env.command_manager.get_term("motion")
        self._processed_actions.add_(command.context[:, 1 : 1 + self.action_dim])


@configclass
class CalibratedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = CalibratedJointPositionAction
