from humanoidverse.utils.torch_utils import *
import torch
from isaac_utils.rotations import wrap_to_pi
from humanoidverse.envs.locomotion.locomotion_ma import LeggedRobotLocomotion

DEBUG = False
class LeggedRobotLocomotionStance(LeggedRobotLocomotion):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.stand_prob = self.config.stand_prob
        self.init_done = True

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 5), dtype=torch.float32, device=self.device
        )
        self.command_ranges = self.config.locomotion_command_ranges
    
    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # Sample the tapping or stand command with a probability
        self.commands[env_ids, 4] = (torch.rand(len(env_ids), device=self.device) > self.stand_prob).float()
        self.commands[env_ids, 0] *= self.commands[env_ids, 4]
        self.commands[env_ids, 1] *= self.commands[env_ids, 4]
        self.commands[env_ids, 2] *= self.commands[env_ids, 4] 
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        self.commands = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands
    
    ################ Curriculum #################
    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Push the robots randomly
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids)
        # Update locomotion commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), 
            self.command_ranges["ang_vel_yaw"][0], 
            self.command_ranges["ang_vel_yaw"][1]
        )
        self.commands[:, 0] *= self.commands[:, 4]
        self.commands[:, 1] *= self.commands[:, 4]
        self.commands[:, 2] *= self.commands[:, 4] 
        
    ######################### PENALTY REWARDS #########################
    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) * self.commands[:, 4] # only apply the base linear z-axis velocity penalty if locomoting
    
    def _reward_near_contact_vel(self):
        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        feet_vel_z = self.simulator._rigid_body_vel[:, self.feet_indices, 2]
        desired_feet_vel_z = -0.05
        rewards = - torch.square(feet_vel_z - desired_feet_vel_z)
        
        rewards = rewards * (feet_height > 0.04)
        rewards = rewards * (feet_height < 0.10)
        rewards = rewards * (feet_vel_z < 0)
        return torch.sum(rewards, dim=1) * (self.commands[:, 4])
    
    def _reward_penalty_contact(self):
        # Initialize the penalty reward tensor
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Check if the agent is in stance mode (commands[:, 4] == 0)
        is_stance = (self.commands[:, 4] == 0)
        # Determine foot contact (contact force in Z-axis > 1 is considered ground contact)
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1  
        # Count the number of feet in contact with the ground
        num_feet_on_ground = contact.sum(dim=1)
        # Penalize if any foot is off the ground when commands[:, 4] == 0 (stance mode)
        res[is_stance & (num_feet_on_ground < 2)] = 1.0  
        # Penalize if both feet are on the ground when commands[:, 4] == 1 (walking mode)
        res[~is_stance & ((num_feet_on_ground == 2) | (num_feet_on_ground == 0))] = 1.0  
        return res
    
    def _reward_penalty_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        contact_feet_vel = feet_vel * contact.unsqueeze(-1)
        penalize = torch.sum(torch.square(contact_feet_vel[:, :, :3]), dim=(1,2))
        stance_envs_id = torch.where(self.commands[:, 4] == 0)[0]
        penalize[stance_envs_id] *= 10.0
        return penalize
    
    def _reward_penalty_shift_in_zero_command(self):
        shift_vel = torch.norm(self.simulator._rigid_body_vel[:, self.pelvis_index, :2], dim=-1) * (torch.norm(self.commands[:, :2], dim=1) < 0.2) * self.commands[:, 4]
        # print(shift_vel)
        return shift_vel

    def _reward_penalty_ang_shift_in_zero_command(self):
        ang_vel = torch.abs(self.simulator._rigid_body_ang_vel[:, self.pelvis_index, 2])  # assuming index 5 = angular z
        # Apply penalty only when there's no angular command (or very low)
        zero_ang_command_mask = (torch.abs(self.commands[:, 2]) < 0.1)
        ang_shift = ang_vel * zero_ang_command_mask * self.commands[:, 4]
        return ang_shift
    #################### STANCE RELATED REWARDS ####################
    
    def _reward_tracking_base_height(self):
        base_height_error = torch.abs(self.config.rewards.desired_base_height - self.simulator.robot_root_states[:, 2])
        # print(base_height_error)
        return torch.exp(-base_height_error/self.config.rewards.reward_tracking_sigma.base_height)*(1.0 - self.commands[:, 4])
    
    def _reward_penalty_stance_dof(self):
        # Penalize the lower body dof velocity
        return torch.sum(torch.square(self.simulator.dof_vel[:, self.lower_body_action_indices]), dim=1) * (1.0 - self.commands[:, 4])
    
    def _reward_penalty_stance_feet(self):
        # Penalize the feet distance on the x axis of base frame
        feet_diff = torch.abs(self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] - self.simulator._rigid_body_pos[:, self.feet_indices[1], :3])
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_index]
        projected_feet_diff = quat_rotate_inverse(pelvis_quat, feet_diff)
        return torch.abs(projected_feet_diff[:, 0]) * (1.0 - self.commands[:, 4])
    
    def _reward_penalty_stance_tap_feet(self):
        # Penalize the feet distance on the x axis of base frame
        feet_diff = torch.abs(self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] - self.simulator._rigid_body_pos[:, self.feet_indices[1], :3])
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_index]
        projected_feet_diff = quat_rotate_inverse(pelvis_quat, feet_diff)
        stance_tap = self.commands[:, 4] * (torch.abs(self.commands[:, 0]) > 0.0)
        return torch.abs(projected_feet_diff[:, 0]) * (1.0 - stance_tap)
    
    def _reward_penalty_stance_root(self):
        # Penalize the root position
        feet_mid_pos = (self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] + self.simulator._rigid_body_pos[:, self.feet_indices[1], :3]) / 2
        root_pos = self.simulator._rigid_body_pos[:, self.pelvis_index, :3]
        root_feet_diff = root_pos - feet_mid_pos
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_index]
        projected_root_feet_diff = quat_rotate_inverse(pelvis_quat, root_feet_diff)
        # return torch.norm(projected_root_feet_diff[:, :2], dim=1) * (1.0 - self.commands[:, 4])
        return torch.abs(projected_root_feet_diff[:, 1]) * (1.0 - self.commands[:, 4])

    def _reward_penalty_stand_still(self):
        # Penalize standing still
        no_contacts = torch.sum(self.simulator.contact_forces[:, self.feet_indices, 2] < 0.1, dim=1) > 0
        return no_contacts.float() * (1.0 - self.commands[:, 4])
    
    def _reward_penalty_stance_symmetry(self):
        # TODO: Hardcoded
        diff_lower_body_dof_po_no = self.simulator.dof_pos[:, self.lower_left_dofs_idx_no] - self.simulator.dof_pos[:, self.lower_right_dofs_idx_no]
        diff_lower_body_dof_pos_op = self.simulator.dof_pos[:, self.lower_left_dofs_idx_op] + self.simulator.dof_pos[:, self.lower_right_dofs_idx_op]
        return torch.sum(torch.abs(diff_lower_body_dof_po_no) + 
                         torch.abs(diff_lower_body_dof_pos_op), dim=1) * (1.0 - self.commands[:, 4])
        
    ######################### Observations #########################
    def _get_obs_command_stand(self):
        return self.commands[:, 4:5]
    