from humanoidverse.utils.torch_utils import *
import torch
from humanoidverse.envs.locomotion.locomotion_ma_stand import LeggedRobotLocomotionStance

class LeggedRobotLocomotionStanceGait(LeggedRobotLocomotionStance):
    def __init__(self, config, device):
        super().__init__(config, device)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.gait_commands = torch.zeros(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self.gait_commands[env_ids, 0] = torch_rand_float(self.command_ranges["gait_period"][0], self.command_ranges["gait_period"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.T = self.gait_commands

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating(command)
        self.gait_commands *= 0.0
        self.gait_commands[:, 0] += 0.65
        self.T = self.gait_commands
    
    def _setup_simulator_control(self):
        super()._setup_simulator_control()
        if self.is_evaluating:
            # In evaluation mode, read gait commands from simulator (set by keyboard input)
            if hasattr(self.simulator, 'gait_commands') and self.simulator.gait_commands is not None:
                self.gait_commands = self.simulator.gait_commands.clone()
                self.T = self.gait_commands
        else:
            # In training mode, write gait commands from environment to simulator
            self.simulator.gait_commands = self.gait_commands

    def update_phase_time(self):
        # Update the phase time
        self.phase_time = self._calc_phase_time()
        self.phase_left = (self.phase_time + self.left_offset) % 1
        self.phase_right = (self.phase_time + self.right_offset) % 1
        self.leg_phase = torch.cat([self.phase_left, self.phase_right], dim=-1)

    ########################## Rewards #########################

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        ratio = self.config.get("single_foot_ratio", 0.55)
        for i in range(2): # left and right feet
            is_stance = (self.leg_phase[:, i] < ratio) | (self.commands[:, 4] == 0)
            contact = self.simulator.contact_forces[:, self.feet_indices[i], 2] > 1
            contact_reward = ~(contact ^ is_stance)
            contact_penalty = contact ^ is_stance
            res += contact_reward.int() - contact_penalty.int()
        return res

    ########################## Observations #########################
    def _get_obs_command_gait(self):
        return self.gait_commands[:, 0:1]
