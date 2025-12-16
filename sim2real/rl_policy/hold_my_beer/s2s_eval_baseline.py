import sys
import numpy as np
import torch
import argparse
import yaml
import os
from termcolor import colored

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy

class S2SEvalPolicy(DecLocomotionPolicy):
    def __init__(self, config, model_path, rl_rate=None, policy_action_scale=0.25, record_joints="upper_body"):
        # Use rl_rate from config if not provided (default 100Hz to match training)
        if rl_rate is None:
            rl_rate = config.get("rl_rate", 100)
        super().__init__(config, model_path, rl_rate, policy_action_scale)
        self.record_joints = record_joints  # "upper_body", "lower_body", or "all"
        
        # --- EE Command Initialization ---
        # Hardcoded default values from training config (g1_27dof_fakehand_ee_rrh.yaml)
        # ee_command layout: [active_prob, x, y, z, tolerance]
        self.num_end_effectors = config.get("num_end_effectors")
        print(f"num_end_effectors: {self.num_end_effectors}")
        self.ee_command = np.zeros((1, 5 * self.num_end_effectors))
        
        # Default offset: [0.3, -0.15, 0.05]
        self.ee_offset = np.array([0.3, -0.15, 0.05])
        self.ee_tolerance = 0.15   #from 0.1 to 0.15 ,bigger tolerance means more loose tracking
        
        # Initialize: active=1, pos=offset, tolerance=0.15
        self.ee_command[0, 0] = 1.0
        self.ee_command[0, 1:4] = self.ee_offset
        self.ee_command[0, 4] = self.ee_tolerance
        
        # Command Gait Period (if not initialized in parent)
        # BasePolicy handles gait_period, but we need it as a command input
        self.gait_command = np.array([[self.gait_period]])
        
        # Initialize stand_command to 1 (walking mode) by default
        # stand_command=0 means stance, stand_command=1 means locomotion
        # If it's 0, phase_time will always be 0, which might confuse the model
        self.stand_command = np.array([[1.0]])  # Start in walking mode (FIXED: was 0.0)
        
        # Initialize last_policy_action to zeros (same as training)
        # Training initializes actions to zeros, so we match that
        self.last_policy_action = np.zeros((1, self.num_dofs))
        
        # Initialize phase time counter (similar to episode_length_buf in training)
        # Training uses: phase_time = (episode_length_buf * dt + phi_offset) % T / T * stand_command
        self.phase_step_counter = 0
        self.phi_offset = 0.0  # Can be randomized like training, but start with 0 for consistency

        self.logger.info("S2S Eval Policy Initialized with EE Control")
        self.logger.info(f"Initial stand_command: {self.stand_command[0, 0]}")
        self.logger.info(f"Initial ee_command: {self.ee_command[0, :]}")
        self.logger.info(f"Initial gait_command: {self.gait_command[0, 0]}")

    def get_current_obs_buffer_dict(self, robot_state_data):
        # Override parent method to include EE and Gait commands
        # Note: We are NOT calling super().get_current_obs_buffer_dict() because 
        # DecLocomotionPolicy adds fields we might not want or in wrong order.
        # Instead we call BasePolicy's method and then add our own.
        
        # 1. Get base observations (base_quat, base_ang_vel, dof_pos, dof_vel, projected_gravity)
        # This calls BasePolicy.get_current_obs_buffer_dict directly (bypassing DecLocomotionPolicy)
        # We can do this by using super(DecLocomotionPolicy, self)
        current_obs_buffer_dict = super(DecLocomotionPolicy, self).get_current_obs_buffer_dict(robot_state_data)
        
        # 2. Add commands
        current_obs_buffer_dict["command_lin_vel"] = self.lin_vel_command
        current_obs_buffer_dict["command_ang_vel"] = self.ang_vel_command
        current_obs_buffer_dict["command_stand"] = self.stand_command
        current_obs_buffer_dict["command_ee"] = self.ee_command
        current_obs_buffer_dict["command_gait"] = self.gait_command
        
        # 3. Add state info
        # Note: new model uses full 27 dof actions history
        current_obs_buffer_dict["actions"] = self.last_policy_action 
        
        # 4. Add Phase info
        # Use parent's phase calculation
        current_obs_buffer_dict["phase_time"] = self._get_obs_phase_time() # Update self.phase_time
        current_obs_buffer_dict["sin_phase"] = np.sin(2 * np.pi * self.phase_time)
        current_obs_buffer_dict["cos_phase"] = np.cos(2 * np.pi * self.phase_time)

        return current_obs_buffer_dict
    
    def _print_obs_structure(self, robot_state_data):
        """Print detailed observation structure and dimensions."""
        # Get current observation buffer
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        
        # Parse observations
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        
        print(f"\n{'='*50}")
        print(f"Sim-to-Real Observation Structure:")
        print(f"{'='*50}")
        
        for obs_key in self.obs_dict:
            obs_keys = sorted(self.obs_dict[obs_key])
            current_obs = current_obs_dict[obs_key]
            
            print(f"\nObservation Group: '{obs_key}'")
            print(f"Single Frame Dimension: {current_obs.shape[-1]}")
            
            start_idx = 0
            for key in obs_keys:
                if key in current_obs_buffer_dict:
                    data = current_obs_buffer_dict[key]
                    # Handle both 1D and 2D arrays: (dim,) or (1, dim)
                    if len(data.shape) == 1:
                        dim = data.shape[0]
                    else:
                        dim = data.shape[-1]
                    scale = self.obs_scales.get(key, 1.0)
                    print(f"  [{start_idx:4d} - {start_idx + dim - 1:4d}] {key:<30} (Dim: {dim:2d}, Scale: {scale:.3f})")
                    start_idx += dim
                else:
                    print(f"  [WARNING] {key} not found in current_obs_buffer_dict")
            
            try:
                h_len = self.history_length_dict[obs_key]
                total_dim = current_obs.shape[-1] * h_len
                print(f"History Length: {h_len}")
                print(f"Total Stacked Dimension: {total_dim}")
            except:
                pass
        
        # Print command values
        print(f"\n{'='*50}")
        print(f"Command Values:")
        print(f"{'='*50}")
        print(f"  lin_vel_command:     {self.lin_vel_command[0]}")
        print(f"  ang_vel_command:     {self.ang_vel_command[0, 0]:.3f}")
        print(f"  stand_command:       {self.stand_command[0, 0]:.1f}")
        print(f"  gait_command:        {self.gait_command[0, 0]:.3f}")
        if hasattr(self, 'ee_command'):
            ee_cmd = self.ee_command[0]
            if len(ee_cmd) >= 5:
                print(f"  ee_command:           active={ee_cmd[0]:.1f}, pos=[{ee_cmd[1]:.3f}, {ee_cmd[2]:.3f}, {ee_cmd[3]:.3f}], tolerance={ee_cmd[4]:.3f}")
            else:
                print(f"  ee_command:           {ee_cmd}")
        print(f"{'='*50}\n")
    
    def rl_inference(self, robot_state_data):
        """Perform RL inference to get policy action."""
        obs = self.prepare_obs_for_rl(robot_state_data)
        
        # Print observation structure on first call
        if not hasattr(self, '_obs_structure_printed'):
            self._print_obs_structure(robot_state_data)
            self._obs_structure_printed = True
        
        # Debug: Print observation shape and some key values
        if not hasattr(self, '_debug_printed'):
            self.logger.info(f"Actor obs shape: {obs['actor_obs'].shape}")
            # obs_dim_dict['actor_obs'] is already an integer (total dimension), not a dict
            expected_dim = self.obs_dim_dict['actor_obs'] * self.history_length_dict['actor_obs']
            self.logger.info(f"Expected shape: (1, {expected_dim})")
            self.logger.info(f"Command lin_vel: {self.lin_vel_command}")
            self.logger.info(f"Command stand: {self.stand_command}")
            self.logger.info(f"Phase time: {self.phase_time}")
            self._debug_printed = True
        
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        # Print action key order and structure (only first time)
        if not hasattr(self, '_action_keys_printed'):
            # Infer body_keys from config or use default
            body_keys = self.config.get("body_keys", ["lower_body", "upper_body"])
            # Get dimensions from config
            dof_names_lower = self.config.get("dof_names_lower_body", [])
            dof_names_upper = self.config.get("dof_names_upper_body", [])
            num_lower_dofs = len(dof_names_lower) if dof_names_lower else 13  # Default to 13 if not found
            num_upper_dofs = len(dof_names_upper) if dof_names_upper else self.config.get("NUM_UPPER_BODY_JOINTS", 14)
            
            print(f"\n{'='*60}")
            print(f"Action Output Structure (S2S Eval):")
            print(f"{'='*60}")
            print(f"Action Keys Order: {body_keys}")
            action_start_idx = 0
            for key in body_keys:
                if key == "lower_body":
                    action_dim = num_lower_dofs
                elif key == "upper_body":
                    action_dim = num_upper_dofs
                else:
                    action_dim = 0
                if action_dim > 0:
                    print(f"  [{action_start_idx:3d} - {action_start_idx + action_dim - 1:3d}] {key:<20} (Dim: {action_dim:2d})")
                    
                    # Print detailed joint information for each body part
                    if key == "lower_body" and dof_names_lower:
                        print(f"    Lower Body Joints:")
                        for i, dof_name in enumerate(dof_names_lower):
                            joint_idx = action_start_idx + i
                            print(f"      [{joint_idx:3d}] {dof_name}")
                    elif key == "upper_body" and dof_names_upper:
                        print(f"    Upper Body Joints:")
                        for i, dof_name in enumerate(dof_names_upper):
                            joint_idx = action_start_idx + i
                            print(f"      [{joint_idx:3d}] {dof_name}")
                    
                    action_start_idx += action_dim
            # Get actual action dimension from policy output
            actual_action_dim = policy_action.shape[-1] if len(policy_action.shape) > 1 else policy_action.shape[0]
            print(f"Total Action Dimension: {actual_action_dim}")
            print(f"Policy Action Shape: {policy_action.shape}")
            print(f"{'='*60}\n")
            self._action_keys_printed = True
        
        # Debug: Print action stats (only first time and every 100 steps)
        if not hasattr(self, '_inference_count'):
            self._inference_count = 0
        self._inference_count += 1
        
        if self._inference_count == 1 or self._inference_count % 100 == 0:
            self.logger.info(f"Policy action range: [{policy_action.min():.3f}, {policy_action.max():.3f}], mean: {policy_action.mean():.3f}")
        
        # Store for history
        self.last_policy_action = policy_action.copy()
        
        # Scale action
        scaled_policy_action = policy_action * self.policy_action_scale
        
        
        return scaled_policy_action


    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states 27 dof data we need
        robot_state_data = self.state_processor.robot_state_data

        if self.get_ready_state:
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)

        elif not self.use_policy_action:
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]

        else:
            scaled_policy_action = self.rl_inference(robot_state_data)
            q_target = scaled_policy_action + self.default_dof_angles
            
        # 4. 安全限位截断 (Clip)
        if self.motor_pos_lower_limit_list is not None and self.motor_pos_upper_limit_list is not None:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)
            # check the output of q_target
        cmd_q = q_target[0]

        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)
    
    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses."""
        # EE Control
        # Mapping: 
        #   x/c: X axis (forward/backward)
        #   v/b: Y axis (left/right, lateral)
        #   n/m: Z axis (up/down, vertical)
        
        step_size = 0.03
        
        if keycode == "x": # X forward
            self.ee_command[0, 1] += step_size
            self.logger.info(f"EE Pos (x,y,z): {self.ee_command[0, 1:4]}")
        elif keycode == "c": # X backward
            self.ee_command[0, 1] -= step_size
            self.logger.info(f"EE Pos (x,y,z): {self.ee_command[0, 1:4]}")
        elif keycode == "v": # Y left (positive Y in body frame, lateral)
            self.ee_command[0, 2] += step_size
            self.logger.info(f"EE Pos (x,y,z): {self.ee_command[0, 1:4]}")
        elif keycode == "b": # Y right
            self.ee_command[0, 2] -= step_size
            self.logger.info(f"EE Pos (x,y,z): {self.ee_command[0, 1:4]}")
        elif keycode == "n": # Z up (vertical)
            self.ee_command[0, 3] += step_size
            self.logger.info(f"EE Pos (x,y,z): {self.ee_command[0, 1:4]}")
        elif keycode == "m": # Z down
            self.ee_command[0, 3] -= step_size
            self.logger.info(f"EE Pos (x,y,z): {self.ee_command[0, 1:4]}")
            
        # Gait Period Control
        elif keycode == "1":
            self.gait_command[0, 0] += 0.05
            self.gait_period = self.gait_command[0, 0]
            self.logger.info(f"Gait Period: {self.gait_period:.2f}")
        elif keycode == "2":
            self.gait_command[0, 0] -= 0.05
            self.gait_period = self.gait_command[0, 0]
            self.logger.info(f"Gait Period: {self.gait_period:.2f}")
        else:
            # Call parent (DecLocomotionPolicy) to handle WASD, QE, Z, etc.
            super().handle_keyboard_button(keycode)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2S Eval")
    # Default to our new config
    parser.add_argument("--config", type=str, default="config/g1/g1_27dof_ee_sim.yaml", help="config file")
    parser.add_argument("--model_path", type=str,default="models/hold_my_beer/baseline_ft_10000.onnx", help="path to the ONNX model file")
    parser.add_argument("--record_joints", type=str, default="upper_body", 
                        choices=["upper_body", "lower_body", "all"],
                        help="Which joints to record: upper_body, lower_body, or all (default: upper_body)")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided")

    # Get rl_rate from config (default 100Hz to match training, or use config value)
    rl_rate = config.get("rl_rate", 100)
    policy = S2SEvalPolicy(config, model_path, rl_rate=rl_rate, policy_action_scale=0.25, record_joints=args.record_joints)
    policy.run()

