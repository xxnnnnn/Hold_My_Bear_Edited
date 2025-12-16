import argparse
import sys
import threading
import time
from threading import Thread
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import glfw
import numpy as np
import yaml
from loguru import logger
from loop_rate_limiters import RateLimiter

sys.path.append("../")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from sim2real.utils.robot import Robot

from sim2real.utils.sdk2py_bridge import ElasticBand, create_sdk2py_bridge


class BaseSimulator:
    def __init__(self, config, save_dir=None):
        self.config = config
        self.save_dir = save_dir
        self.init_config()
        self.init_scene()
        self.init_factory()
        self.init_robot_bridge()
        self.init_recording()

        self.sim_thread = Thread(target=self.simulation_thread)

    def init_config(self):
        self.robot = Robot(self.config)
        self.sdk_type = self.config.get("SDK_TYPE", "unitree")
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)
        self.logger = logger
        self.rate = RateLimiter(1 / self.config["SIMULATE_DT"],warn=False)

    def init_factory(self):
        if self.sdk_type == "unitree":
            if self.config.get("INTERFACE", None):
                if sys.platform == "linux":
                    self.config["INTERFACE"] = "lo"
                elif sys.platform == "darwin":
                    self.config["INTERFACE"] = "lo0"
                else:
                    raise NotImplementedError("Only support Linux and MacOS.")
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        elif self.sdk_type == "booster":
            from booster_robotics_sdk_python import ChannelFactory

            ChannelFactory.Instance().Init(self.config["DOMAIN_ID"])
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")
        self.logger.info(str.format("SDK TYPE: {0}", self.sdk_type))

    def init_scene(self):
        print(self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
        self.base_id = self.mj_model.body(base_body_name).id

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            band_attached_link_name = self.config.get("BAND_ATTACHED_LINK", "torso_link")
            self.band_attached_link = self.mj_model.body(band_attached_link_name).id
            # Create combined key callback
            def combined_key_callback(key):
                self.elastic_band.MujuocoKeyCallback(key)
                self._key_callback(key)
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=combined_key_callback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self._key_callback
            )

    def init_robot_bridge(self):
        self.robot_bridge = create_sdk2py_bridge(self.mj_model, self.mj_data, self.config)
        if self.config["USE_JOYSTICK"]:
            if sys.platform == "linux":  # TODO [Yuanhang]: add other joystick support
                if self.config["SDK_TYPE"] == "unitree":
                    self.robot_bridge.SetupJoystick(
                        device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
                    )
                else:
                    self.logger.warning(f"Joystick is not supported for {self.config['SDK_TYPE']} yet.")
            else:
                self.logger.warning("Joystick is not supported on Windows or MacOS.")
    
    def init_recording(self):
        """Initialize recording functionality for right wrist joint accelerations."""
        self.is_recording = False
        self.recorded_data = {
            'wrist_roll_acc': [],
            'wrist_pitch_acc': [],
            'wrist_yaw_acc': [],
            'wrist_total_acc_magnitude': []
        }
        
        # Find right wrist joint motor indices from config
        # Use WeakMotorJointIndex to get motor indices, then use qacc[6 + motor_idx]
        right_wrist_joint_names = [
            'right_wrist_roll_joint',
            'right_wrist_pitch_joint',
            'right_wrist_yaw_joint'
        ]
        
        joint2motor = self.config.get("WeakMotorJointIndex", {})
        self.right_wrist_motor_indices = []
        
        for joint_name in right_wrist_joint_names:
            if joint_name in joint2motor:
                motor_idx = joint2motor[joint_name]
                self.right_wrist_motor_indices.append(motor_idx)
                self.logger.info(f"Found joint {joint_name} at motor index {motor_idx}")
            else:
                self.logger.warning(f"Joint {joint_name} not found in WeakMotorJointIndex")
        
        if len(self.right_wrist_motor_indices) != 3:
            self.logger.error(f"Failed to find all right wrist joints. Found motor indices: {self.right_wrist_motor_indices}")
            self.right_wrist_motor_indices = []  # Disable recording if joints not found
        else:
            self.logger.info(f"Right wrist joint motor indices: {self.right_wrist_motor_indices}")
    
    def _key_callback(self, key):
        """Handle keyboard key press for recording toggle."""
        if key == glfw.KEY_R:
            if not self.is_recording:
                # Start recording
                self.is_recording = True
                self.recorded_data = {
                    'wrist_roll_acc': [],
                    'wrist_pitch_acc': [],
                    'wrist_yaw_acc': [],
                    'wrist_total_acc_magnitude': []
                }
                self.logger.info("Started recording right wrist accelerations")
            else:
                # Stop recording and save
                self.is_recording = False
                self._save_recorded_data()
                self.logger.info("Stopped recording and saved data")
    
    def _save_recorded_data(self):
        """Save recorded data to npz file."""
        if not self.recorded_data['wrist_roll_acc']:
            self.logger.warning("No data recorded, skipping save")
            return
        
        # Determine save directory
        if self.save_dir:
            save_path = Path(self.save_dir)
        else:
            # Auto-increment exp folder (save under ./result)
            base_dir = Path("result")
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Find highest existing exp number
            max_exp_num = 0
            for item in base_dir.iterdir():
                if item.is_dir() and item.name.startswith("exp"):
                    try:
                        exp_num = int(item.name[3:])
                        max_exp_num = max(max_exp_num, exp_num)
                    except ValueError:
                        continue
            
            # Create next exp folder
            next_exp_num = max_exp_num + 1
            save_path = base_dir / f"exp{next_exp_num}"
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_path / f"wrist_acc_{timestamp}.npz"
        
        # Convert lists to numpy arrays
        save_dict = {
            'wrist_roll_acc': np.array(self.recorded_data['wrist_roll_acc']),
            'wrist_pitch_acc': np.array(self.recorded_data['wrist_pitch_acc']),
            'wrist_yaw_acc': np.array(self.recorded_data['wrist_yaw_acc']),
            'wrist_total_acc_magnitude': np.array(self.recorded_data['wrist_total_acc_magnitude'])
        }
        
        np.savez(filename, **save_dict)
        self.logger.info(f"Saved recorded data to {filename}")

    def compute_torques(self):
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)
            try:
                for i in range(self.robot_bridge.num_motor):    
                    self.torques[i] = (
                        motor_cmd[i].tau
                        + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[7 + i])
                        + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[6 + i])
                    )
                   
            except Exception as e:
                self.logger.error(str.format("Joint {0} not found in motor_cmd: {1}", i, e))
        # Set the torque limit
        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
        self.compute_torques()
        if self.robot_bridge.free_base:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
        
        # Record wrist accelerations if recording
        if self.is_recording and len(self.right_wrist_motor_indices) == 3:
            # Get joint accelerations from qacc
            # If free_base, qacc[0:6] is base, qacc[6:] corresponds to motor indices
            # Use the same approach as unitree_sdk2py_bridge: qacc[6 + motor_idx]
            qacc_offset = 6 if self.robot_bridge.free_base else 0
            
            wrist_accs = []
            for motor_idx in self.right_wrist_motor_indices:
                qacc_idx = qacc_offset + motor_idx
                if qacc_idx < len(self.mj_data.qacc):
                    acc = self.mj_data.qacc[qacc_idx]
                    wrist_accs.append(acc)
                else:
                    self.logger.warning(f"qacc index {qacc_idx} out of range (len={len(self.mj_data.qacc)}, motor_idx={motor_idx})")
                    wrist_accs.append(0.0)
            
            if len(wrist_accs) == 3:
                self.recorded_data['wrist_roll_acc'].append(wrist_accs[0])
                self.recorded_data['wrist_pitch_acc'].append(wrist_accs[1])
                self.recorded_data['wrist_yaw_acc'].append(wrist_accs[2])
                # Calculate magnitude
                magnitude = np.linalg.norm(wrist_accs)
                self.recorded_data['wrist_total_acc_magnitude'].append(magnitude)

    def simulation_thread(self):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.logger.info(str.format("FPS: {0:.2f}", 100 / (end_time - start_time)))
                start_time = end_time
            self.rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_27dof_ee_sim.yaml", help="config file")
    parser.add_argument("--save_dir", type=str, default=None, 
                        help="Directory to save recorded data. If not specified, auto-increments exp1, exp2, etc. in ./result/")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = BaseSimulator(config, save_dir=args.save_dir)
    simulation.sim_thread.start()
