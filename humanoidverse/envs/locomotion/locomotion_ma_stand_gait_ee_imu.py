from humanoidverse.utils.torch_utils import *
import torch
from humanoidverse.envs.locomotion.locomotion_ma_stand_gait_ee import LeggedRobotLocomotionStanceGaitEETracking

# DEBUG = False
class LeggedRobotLocomotionStanceGaitEETrackingIMU(LeggedRobotLocomotionStanceGaitEETracking):
    def __init__(self, config, device):
        super().__init__(config, device)
        
    def _init_buffers(self):
        super()._init_buffers()
        # IMU index: which end effector to use as IMU (default to first one, e.g., rrh)
        self.imu_ee_index = self.config.get("imu_ee_index", 0)
        
        # IMU buffers: accelerometer and gyroscope readings in local frame
        # Shape: (num_envs, 3) - only use one end effector for IMU
        self.ee_imu_acc = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ee_imu_gyro = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        
        # Buffer for previous velocity to compute acceleration
        self.last_imu_ee_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        
        # Gravity vector in world frame (pointing down)
        self.world_gravity = torch.tensor([0., 0., -9.81], dtype=torch.float32, device=self.device)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)
        if target_buf is not None:
            if "ee_imu_acc" in target_buf:
                self.ee_imu_acc[env_ids] = target_buf["ee_imu_acc"].to(self.ee_imu_acc.dtype)
            if "ee_imu_gyro" in target_buf:
                self.ee_imu_gyro[env_ids] = target_buf["ee_imu_gyro"].to(self.ee_imu_gyro.dtype)
            if "last_imu_ee_vel" in target_buf:
                self.last_imu_ee_vel[env_ids] = target_buf["last_imu_ee_vel"].to(self.last_imu_ee_vel.dtype)
        else:
            self.ee_imu_acc[env_ids] = 0.
            self.ee_imu_gyro[env_ids] = 0.
            self.last_imu_ee_vel[env_ids] = 0.

    def _post_compute_observations_callback(self):
        super()._post_compute_observations_callback()
        
        # Get the IMU end effector's current state
        # Use the specified end effector index for IMU simulation
        imu_ee_vel = self.end_effector_vel[:, self.imu_ee_index, :]  # (num_envs, 3)
        imu_ee_ang_vel = self.end_effector_ang_vel[:, self.imu_ee_index, :]  # (num_envs, 3)
        imu_ee_rot = self.end_effector_rot[:, self.imu_ee_index, :]  # (num_envs, 4)
        
        # Compute acceleration in world frame
        world_acc = (imu_ee_vel - self.last_imu_ee_vel) / self.dt  # (num_envs, 3)
        
        # IMU gravity compensation
        proper_acc_world = world_acc - self.world_gravity  # Subtract gravity (accelerometer at rest reads +g)
        
        # Transform to local frame using quat_rotate_inverse
        self.ee_imu_acc = quat_rotate_inverse(imu_ee_rot, proper_acc_world)
        
        # Gyroscope: angular velocity in local frame
        self.ee_imu_gyro = quat_rotate_inverse(imu_ee_rot, imu_ee_ang_vel)
        
        # Update last velocity for next iteration
        self.last_imu_ee_vel[:] = imu_ee_vel

    ########################### Observations ###########################
    def _get_obs_ee_imu_acc(self):
        """Get IMU accelerometer reading (linear acceleration in local frame, includes gravity)."""
        return self.ee_imu_acc
    
    def _get_obs_ee_imu_gyro(self):
        """Get IMU gyroscope reading (angular velocity in local frame)."""
        return self.ee_imu_gyro
