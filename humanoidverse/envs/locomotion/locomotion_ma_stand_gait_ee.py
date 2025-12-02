from humanoidverse.utils.torch_utils import *
import torch
from humanoidverse.envs.locomotion.locomotion_ma_stand_gait import LeggedRobotLocomotionStanceGait

# DEBUG = False
class LeggedRobotLocomotionStanceGaitEETracking(LeggedRobotLocomotionStanceGait):
    def __init__(self, config, device):
        super().__init__(config, device)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.end_effector_name = self.config.robot.end_effector_name
        self.end_effector_index = [self.simulator.find_rigid_body_indice(name) for name in self.end_effector_name]
        self.num_end_effectors = self.config.robot.num_end_effectors
        assert len(self.end_effector_index) == self.num_end_effectors, f"End effector names {self.end_effector_name} not found in the robot model"
        
        self.ee_commands = torch.zeros(
            (self.num_envs, 5 * self.num_end_effectors), dtype=torch.float32, device=self.device
        )
        self.ee_command_ranges = self.config.ee_command_ranges
        self.end_effector_pos = self.simulator._rigid_body_pos[:, self.end_effector_index]
        self.end_effector_rot = self.simulator._rigid_body_rot[:, self.end_effector_index, :]
        self.end_effector_rot_gravity = quat_rotate_inverse(self.end_effector_rot.reshape(-1, 4), self.gravity_vec.repeat(self.num_end_effectors, 1)).reshape(-1, self.num_end_effectors, 3)

        self.end_effector_vel = self.simulator._rigid_body_vel[:, self.end_effector_index, :]
        self.end_effector_ang_vel = self.simulator._rigid_body_ang_vel[:, self.end_effector_index, :]
        self.last_end_effector_vel = torch.zeros_like(self.end_effector_vel)
        self.last_end_effector_ang_vel = torch.zeros_like(self.end_effector_ang_vel)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        sync = self.config.ee_command_ranges.get("sync", 0)
        # yitang: 0: normal sampling, 1: sync sampling
        if sync == 1:
            # first ee command as template
            ee0_name = self.end_effector_name[0]
            active = (torch.rand(1, device=self.device) < self.config.ee_command_ranges[ee0_name]["active_prob"]).float()
            x = torch_rand_float(
                self.ee_command_ranges[ee0_name]["x"][0],
                self.ee_command_ranges[ee0_name]["x"][1],
                (1, 1), device=self.device
            )
            y = torch_rand_float(
                self.ee_command_ranges[ee0_name]["y"][0],
                self.ee_command_ranges[ee0_name]["y"][1],
                (1, 1), device=self.device
            )
            z = torch_rand_float(
                self.ee_command_ranges[ee0_name]["z"][0],
                self.ee_command_ranges[ee0_name]["z"][1],
                (1, 1), device=self.device
            )
            tolerance = torch_rand_float(
                self.ee_command_ranges[ee0_name]["tolerance"][0],
                self.ee_command_ranges[ee0_name]["tolerance"][1],
                (1, 1), device=self.device
            )

        for ee_idx in range(self.num_end_effectors):
            ee_name = self.end_effector_name[ee_idx]
            if sync == 1:
                # use ee0 command and offset for each ee
                offset = torch.tensor(self.ee_command_ranges[ee_name]["offset"], device=self.device)
                self.ee_commands[env_ids, 5 * ee_idx] = active.expand(len(env_ids))
                self.ee_commands[env_ids, (5 * ee_idx + 1):(5 * ee_idx + 2)] = x.expand(len(env_ids), 1) + offset[0]
                self.ee_commands[env_ids, (5 * ee_idx + 2):(5 * ee_idx + 3)] = -y.expand(len(env_ids), 1) + offset[1]
                self.ee_commands[env_ids, (5 * ee_idx + 3):(5 * ee_idx + 4)] = z.expand(len(env_ids), 1) + offset[2]
                self.ee_commands[env_ids, (5 * ee_idx + 4):(5 * ee_idx + 5)] = tolerance.expand(len(env_ids), 1)
            else:
                # random sampling for each ee
                self.ee_commands[env_ids, 5 * ee_idx] = (torch.rand(len(env_ids), device=self.device) < self.config.ee_command_ranges[ee_name]["active_prob"]).float()
                self.ee_commands[env_ids, (5 * ee_idx + 1):(5 * ee_idx + 2)] = torch_rand_float(
                    self.ee_command_ranges[ee_name]["x"][0],
                    self.ee_command_ranges[ee_name]["x"][1],
                    (len(env_ids), 1), device=self.device,
                )
                self.ee_commands[env_ids, (5 * ee_idx + 2):(5 * ee_idx + 3)] = torch_rand_float(
                    self.ee_command_ranges[ee_name]["y"][0],
                    self.ee_command_ranges[ee_name]["y"][1],
                    (len(env_ids), 1), device=self.device,
                )
                self.ee_commands[env_ids, (5 * ee_idx + 3):(5 * ee_idx + 4)] = torch_rand_float(
                    self.ee_command_ranges[ee_name]["z"][0],
                    self.ee_command_ranges[ee_name]["z"][1],
                    (len(env_ids), 1), device=self.device,
                )
                self.ee_commands[env_ids, (5 * ee_idx + 4):(5 * ee_idx + 5)] = torch_rand_float(
                    self.ee_command_ranges[ee_name]["tolerance"][0],
                    self.ee_command_ranges[ee_name]["tolerance"][1],
                    (len(env_ids), 1), device=self.device,
                )
                offset = torch.tensor(self.ee_command_ranges[ee_name]["offset"], device=self.device)
                self.ee_commands[env_ids, (5*ee_idx+1):(5*ee_idx+4)] += offset
    
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating(command)
        self.ee_commands = torch.zeros((self.num_envs, 5 * self.num_end_effectors), dtype=torch.float32, device=self.device)
        for ee_idx in range(self.num_end_effectors):
            ee_name = self.end_effector_name[ee_idx]
            offset = torch.tensor(self.ee_command_ranges[ee_name]["offset"], device=self.device)
            self.ee_commands[:, 5*ee_idx] = 1
            self.ee_commands[:, (5*ee_idx+1):(5*ee_idx+4)] += offset
            self.ee_commands[:, 5 * ee_idx + 4] = self.config.ee_command_ranges[ee_name]["tolerance"][1]
    
    def _setup_simulator_control(self):
        super()._setup_simulator_control()
        if self.is_evaluating:
            # In evaluation mode, read ee commands from simulator (set by keyboard input)
            if hasattr(self.simulator, 'ee_commands') and self.simulator.ee_commands is not None:
                self.ee_commands = self.simulator.ee_commands.clone()
            if hasattr(self.simulator, 'gait_commands') and self.simulator.gait_commands is not None:
                self.gait_commands = self.simulator.gait_commands.clone()
                self.T = self.gait_commands
        else:
            # In training mode, write ee commands from environment to simulator
            self.simulator.ee_commands = self.ee_commands

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)
        if target_buf is not None:
            self.end_effector_vel[env_ids] = target_buf["end_effector_vel"].to(self.end_effector_vel.dtype)
            self.end_effector_ang_vel[env_ids] = target_buf["end_effector_ang_vel"].to(self.end_effector_ang_vel.dtype)
            self.last_end_effector_vel[env_ids] = target_buf["pre_end_effector_vel"].to(self.pre_end_effector_vel.dtype)
            self.last_end_effector_ang_vel[env_ids] = target_buf["pre_end_effector_ang_vel"].to(self.pre_end_effector_ang_vel.dtype)
            self.end_effector_pos[env_ids] = target_buf["end_effector_pos"].to(self.end_effector_pos.dtype)
            self.end_effector_rot[env_ids] = target_buf["end_effector_rot"].to(self.end_effector_rot.dtype)
            self.end_effector_rot_gravity[env_ids] = target_buf["end_effector_rot_gravity"].to(self.end_effector_rot_gravity.dtype)
        else:
            self.end_effector_vel[env_ids] = 0.
            self.end_effector_ang_vel[env_ids] = 0.
            self.last_end_effector_vel[env_ids] = 0.
            self.last_end_effector_ang_vel[env_ids] = 0.
            self.end_effector_pos[env_ids] = 0.
            self.end_effector_rot[env_ids] = 0.
            self.end_effector_rot_gravity[env_ids] = 0.
            
    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        # In evaluation mode, read commands from simulator (set by keyboard input)
        # This needs to be done before computing observations so that commands are available
        if self.is_evaluating:
            if hasattr(self.simulator, 'ee_commands') and self.simulator.ee_commands is not None:
                self.ee_commands = self.simulator.ee_commands.clone()
            if hasattr(self.simulator, 'gait_commands') and self.simulator.gait_commands is not None:
                self.gait_commands = self.simulator.gait_commands.clone()
                self.T = self.gait_commands
        self.end_effector_vel[:] = self.simulator._rigid_body_vel[:, self.end_effector_index, :]
        self.end_effector_ang_vel[:] = self.simulator._rigid_body_ang_vel[:, self.end_effector_index, :]
        self.end_effector_pos = self.simulator._rigid_body_pos[:, self.end_effector_index]
        self.end_effector_rot[:] = self.simulator._rigid_body_rot[:, self.end_effector_index, :]    
        self.end_effector_rot_gravity[:] = quat_rotate_inverse(self.end_effector_rot.reshape(-1, 4), 
                                                            self.gravity_vec.unsqueeze(1).expand(self.num_envs, self.num_end_effectors, 3).reshape(-1, 3)).reshape(-1, self.num_end_effectors, 3)

    def _post_compute_observations_callback(self):
        super()._post_compute_observations_callback()
        self.end_effector_acc = (self.end_effector_vel - self.last_end_effector_vel) / self.dt
        self.end_effector_ang_acc = (self.end_effector_ang_vel - self.last_end_effector_ang_vel) / self.dt
        self.last_end_effector_vel[:] = self.end_effector_vel[:]
        self.last_end_effector_ang_vel[:] = self.end_effector_ang_vel[:]

    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.rand_apply_force = self.config.domain_rand.get("rand_apply_force", False)
        self.rand_apply_force_pos = self.config.domain_rand.get("rand_apply_force_pos", False)
        self.apply_force_tensor = torch.zeros(self.num_envs, self.config.robot.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.apply_force_pos_tensor = torch.zeros(self.num_envs, self.config.robot.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
    def _episodic_domain_randomization(self, env_ids):
        super()._episodic_domain_randomization(env_ids)
        
        if self.rand_apply_force:
            self.domain_rand_ee_force = torch.zeros(self.num_envs, self.num_end_effectors, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.domain_rand_ee_force[env_ids][..., 0] = torch_rand_float(self.config.domain_rand.x_force_range[0], self.config.domain_rand.x_force_range[1], (len(env_ids), self.num_end_effectors), device=self.device)
            self.domain_rand_ee_force[env_ids][..., 1] = torch_rand_float(self.config.domain_rand.y_force_range[0], self.config.domain_rand.y_force_range[1], (len(env_ids), self.num_end_effectors), device=self.device)
            self.domain_rand_ee_force[env_ids][..., 2] = torch_rand_float(self.config.domain_rand.z_force_range[0], self.config.domain_rand.z_force_range[1], (len(env_ids), self.num_end_effectors), device=self.device)

    def _apply_force_in_physics_step(self):
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        if self.rand_apply_force:
            if self.is_evaluating:
                self.domain_rand_ee_force *= 0.0
                self.domain_rand_ee_force[..., 2] = -5
                pass
            self.apply_force_pos_tensor[:, self.end_effector_index,:] = self.simulator._rigid_body_pos[:, self.end_effector_index, :]
            if self.rand_apply_force_pos:
                std = self.config.domain_rand.get("apply_force_pos_std", 0.0)
                noise = (torch.rand_like(self.apply_force_pos_tensor[:, self.end_effector_index, :2]) - 0.5) * std * 2
                self.apply_force_pos_tensor[:, self.end_effector_index, :2] += noise
            self.apply_force_tensor[:, self.end_effector_index, :] = self.domain_rand_ee_force
            self.simulator.apply_rigid_body_force_at_pos_tensor(self.apply_force_tensor, self.apply_force_pos_tensor)
        
        self.simulator.apply_torques_at_dof(self.torques)

    ############################ Curriculum #############################

    ########################### FEET REWARDS ###########################
    def _reward_tracking_end_effector_pos(self):
        #ee_local_pos = ee's global position - pelvis's global position
        ee_local_pos = self.end_effector_pos - self.simulator._rigid_body_pos[:, self.pelvis_index, :3].unsqueeze(1) 
        # rotate ee_local_pos to the base frame
        ee_relative_pos = quat_rotate_inverse(
                            self.base_quat.unsqueeze(1).expand(self.num_envs, self.num_end_effectors, 4).reshape(-1, 4), 
                            ee_local_pos.reshape(-1, 3)).reshape(self.num_envs, self.num_end_effectors, 3)
        
        ee_target_pos = self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 1:4].clone()

        ee_target_pos[:, :, 2] += - self.simulator._rigid_body_pos[:, self.pelvis_index, 2].unsqueeze(1) + self.config.rewards.desired_base_height

        ee_position_error = torch.sum(torch.square(ee_target_pos - ee_relative_pos), dim=2)
        ee_tolerance = self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 4].clone() 
        ee_track_reward = torch.exp(-ee_position_error / (ee_tolerance ** 2))

        return torch.sum(ee_track_reward, dim=-1) / self.num_end_effectors * (1 - self.commands[:, 4]) # only apply the end effector position tracking reward if standing

    def _reward_soft_tracking_end_effector_pos(self):
        ee_local_pos = self.end_effector_pos - self.simulator._rigid_body_pos[:, self.pelvis_index, :3].unsqueeze(1)
        ee_relative_pos = quat_rotate_inverse(
                            self.base_quat.unsqueeze(1).expand(self.num_envs, self.num_end_effectors, 4).reshape(-1, 4), 
                            ee_local_pos.reshape(-1, 3)).reshape(self.num_envs, self.num_end_effectors, 3)
        ee_target_pos = self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 1:4].clone()
        ee_target_pos[:, :, 2] += - self.simulator._rigid_body_pos[:, self.pelvis_index, 2].unsqueeze(1) + self.config.rewards.desired_base_height

        ee_position_error = torch.sum(torch.square(ee_target_pos - ee_relative_pos), dim=2) 
        ee_tolerance = (self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 4] * self.config.rewards.soft_tracking_factor).clone() 
        ee_position_error = torch.where(ee_position_error < 0.05, torch.zeros_like(ee_position_error), ee_position_error)
        ee_track_reward = torch.exp(-ee_position_error / (ee_tolerance ** 2)) 
        return torch.sum(ee_track_reward, dim=-1) / self.num_end_effectors * self.commands[:, 4] # only apply the end effector position tracking reward if locomoting

    def _reward_penalty_end_effector_acc(self):
        # Penalize the end effector acceleration
        end_effector_acc = (self.end_effector_vel - self.last_end_effector_vel) / self.dt
        end_effector_acc_norm = torch.norm(end_effector_acc, dim=2)
        ee_active = self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 0].clone()
        active_counts = ee_active.sum(dim=1).clamp(min=1)
        end_effector_acc_norm = (end_effector_acc_norm * ee_active).sum(dim=1) / active_counts
        return end_effector_acc_norm 

    def _reward_penalty_end_effector_ang_acc(self):
        # Penalize the end effector angular acceleration
        end_effector_ang_acc = (self.end_effector_ang_vel - self.last_end_effector_ang_vel) / self.dt
        end_effector_ang_acc_norm = torch.norm(end_effector_ang_acc, dim=2)
        ee_active = self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 0].clone()
        active_counts = ee_active.sum(dim=1).clamp(min=1)
        end_effector_ang_acc_norm = (end_effector_ang_acc_norm * ee_active).sum(dim=1) / active_counts
        return end_effector_ang_acc_norm 

    def _reward_penalty_end_effector_tilt(self):
        # Penalize the end effector tilt
        end_effector_grav_xy = torch.norm(self.end_effector_rot_gravity[:, :, :2], dim=2)
        end_effector_grav_xy = torch.sum(end_effector_grav_xy, dim=1) / self.num_end_effectors
        return end_effector_grav_xy

    def _reward_partial_upperbody_joint_angle_freeze(self):
        assert self.config.robot.has_upper_body_dof
        # Yitang's Hardcoded
        self.config.robot.upper_body_freeze_dof_indices = self.config.robot.get("upper_body_freeze_dof_indices", [19, 26])
        upper_body_freeze_index = self.config.robot.upper_body_freeze_dof_indices
        diff_freeze = torch.abs(self.simulator.dof_pos[:, upper_body_freeze_index] - self.default_dof_pos[:,upper_body_freeze_index])
        return torch.sum(diff_freeze, dim=1)

    def _reward_penalty_end_effector_acc_exp(self):
        end_effector_acc = (self.end_effector_vel - self.last_end_effector_vel) / self.dt
        end_effector_acc_norm = torch.norm(end_effector_acc, dim=2)
        ee_active = self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 0].clone()
        active_counts = ee_active.sum(dim=1).clamp(min=1)
        end_effector_acc_norm = (end_effector_acc_norm * ee_active).sum(dim=1) / active_counts
        # end_effector_acc_norm = torch.sum(end_effector_acc_norm, dim=1) / self.num_end_effectors
        # print(end_effector_acc_norm)
        return torch.exp(-end_effector_acc_norm / self.config.rewards.ee_state_sigma.acc) 

    def _reward_penalty_end_effector_ang_acc_exp(self):
        end_effector_ang_acc = (self.end_effector_ang_vel - self.last_end_effector_ang_vel) / self.dt
        end_effector_ang_acc_norm = torch.norm(end_effector_ang_acc, dim=2)
        ee_active = self.ee_commands.reshape(self.num_envs, self.num_end_effectors, 5)[:, :, 0].clone()
        active_counts = ee_active.sum(dim=1).clamp(min=1)
        end_effector_ang_acc_norm = (end_effector_ang_acc_norm * ee_active).sum(dim=1) / active_counts
        # end_effector_ang_acc_norm = torch.sum(end_effector_ang_acc_norm, dim=1) / self.num_end_effectors
        # print(end_effector_ang_acc_norm)
        return torch.exp(-end_effector_ang_acc_norm / self.config.rewards.ee_state_sigma.ang_acc)
    
    ########################### Observations ###########################
    def _get_obs_end_effector_relative_pos(self):
        ee_local_pos = self.end_effector_pos - self.simulator._rigid_body_pos[:, self.pelvis_index, :3].unsqueeze(1)
        ee_relative_pos = quat_rotate_inverse(
                            self.base_quat.unsqueeze(1).expand(self.num_envs,
                            self.num_end_effectors, 4).reshape(-1, 4), ee_local_pos.reshape(-1, 3)).reshape(self.num_envs, self.num_end_effectors, 3)
        return ee_relative_pos.reshape(self.num_envs, self.num_end_effectors * 3)

    def _get_obs_end_effector_gravity(self):
        ee_gravity = self.end_effector_rot_gravity.reshape(self.num_envs, self.num_end_effectors, 3)
        return ee_gravity.reshape(self.num_envs, self.num_end_effectors * 3)
    
    def _get_obs_command_ee(self):
        return self.ee_commands
    
    def _draw_debug_vis(self):
        """Draw visualization for end effector origin position."""
        super()._draw_debug_vis()  # Call parent to draw other debug visualizations
        
        # Draw sphere at end effector origin for each environment
        if hasattr(self, 'end_effector_pos') and self.end_effector_pos is not None:
            # end_effector_pos shape: (num_envs, num_end_effectors, 3)
            for env_id in range(min(self.num_envs, 10)):  # Limit to first 10 envs for performance
                for ee_idx in range(self.num_end_effectors):
                    ee_pos = self.end_effector_pos[env_id, ee_idx].cpu().numpy()
                    # Draw a small red sphere at the end effector origin
                    self.simulator.draw_sphere(
                        pos=ee_pos,
                        radius=0.01,  # 1cm radius
                        color=(1.0, 0.0, 0.0),  # Red color
                        env_id=env_id
                    )