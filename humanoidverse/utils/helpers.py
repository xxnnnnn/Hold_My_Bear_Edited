import os
import copy
import torch
from torch import nn
import numpy as np
import random

from typing import Any, List, Dict
from termcolor import colored
from loguru import logger

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def pre_process_config(config) -> None:
    """ Pre-process the config for PPO training
    """
    
    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # convert obs_dims to list of dicts
    each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
    config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {each_dict_obs_dims}")
    auxiliary_obs_dims = {}
    if hasattr(config.env.config.obs, 'obs_auxiliary'):
        _aux_obs_key_list = config.env.config.obs.obs_auxiliary
        auxiliary_obs_dims = {}
        for aux_obs_key, aux_config in _aux_obs_key_list.items():
            auxiliary_obs_dims[aux_obs_key] = 0
            for _key, _num in aux_config.items():
                assert _key in config.env.config.obs.obs_dims.keys()
                auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
        logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"): key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys(): 
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            elif key in auxiliary_obs_dims.keys():
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
            else:
                logger.error(f"{obs_key}: {key} not found in obs_dims")
                raise ValueError(f"{obs_key}: {key} not found in obs_dims")
    config.robot.algo_obs_dim_dict = obs_dim_dict
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

def parse_observation(cls: Any, 
                      key_list: List, 
                      buf_dict: Dict, 
                      obs_scales: Dict, 
                      noise_scales: Dict,
                      current_noise_curriculum_value: Any=1.0) -> None:
    """ Parse observations for the legged_robot_base class
    """
    # Counter for printing observation values (print every 100 observation computation cycles)
    if not hasattr(parse_observation, '_print_counter'):
        parse_observation._print_counter = 0
    
    # Check if we should print (only check once per function call, at the beginning)
    should_print = False
    if 'ee_imu_acc' in key_list or 'ee_imu_gyro' in key_list:
        parse_observation._print_counter += 1
        should_print = (parse_observation._print_counter % 100 == 0)  # Print every 100 steps

    # Store observation statistics for printing
    obs_stats = {}

    for obs_key in key_list:
        if obs_key.endswith("_raw"):
            obs_key = obs_key[:-4]
            obs_noise = 0.
        else:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value

        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]
        
        # Generate noise
        noise = (torch.rand_like(actor_obs) * 2. - 1.) * obs_noise
        noisy_obs = actor_obs + noise
        buf_dict[obs_key] = noisy_obs * obs_scale
        
        # Store statistics for all observations
        if should_print:
            true_val_mean = actor_obs.mean().item()
            true_val_std = actor_obs.std().item()
            true_val_min = actor_obs.min().item()
            true_val_max = actor_obs.max().item()
            scaled_val_mean = buf_dict[obs_key].mean().item()
            scaled_val_std = buf_dict[obs_key].std().item()
            
            obs_stats[obs_key] = {
                'true_mean': true_val_mean,
                'true_std': true_val_std,
                'true_min': true_val_min,
                'true_max': true_val_max,
                'scaled_mean': scaled_val_mean,
                'scaled_std': scaled_val_std,
                'scale': obs_scale,
                'noise_scale': obs_noise,
            }
        
        # Print detailed IMU values for debugging (only for first environment)
        if should_print and obs_key in ['ee_imu_acc', 'ee_imu_gyro']:
            env_id = 0  # Print first environment only
            true_val = actor_obs[env_id].cpu().numpy()
            noise_val = noise[env_id].cpu().numpy()
            noisy_val = noisy_obs[env_id].cpu().numpy()
            scaled_val = buf_dict[obs_key][env_id].cpu().numpy()
            
            print(f"\n[IMU Debug] {obs_key} (env_id={env_id}, step={parse_observation._print_counter}):")
            print(f"  真值 (true):        {true_val}")
            print(f"  噪声 (noise):       {noise_val}")
            print(f"  加噪后 (noisy):     {noisy_val}")
            print(f"  最终值 (scaled):    {scaled_val} (scale={obs_scale})")
            print(f"  噪声尺度 (noise_scale): {obs_noise:.4f}")
    
    # Print statistics for all observations
    if should_print and len(obs_stats) > 0:
        print(f"\n{'='*80}")
        print(f"[Observation Statistics] Step={parse_observation._print_counter} (all envs, mean/std/min/max):")
        print(f"{'='*80}")
        print(f"{'Observation':<25} {'True Mean':<12} {'True Std':<12} {'True Range':<20} {'Scaled Mean':<12} {'Scale':<8} {'Noise':<8}")
        print(f"{'-'*80}")
        for obs_key in sorted(obs_stats.keys()):
            stats = obs_stats[obs_key]
            true_range = f"[{stats['true_min']:.3f}, {stats['true_max']:.3f}]"
            print(f"{obs_key:<25} {stats['true_mean']:>11.3f} {stats['true_std']:>11.3f} {true_range:<20} "
                  f"{stats['scaled_mean']:>11.3f} {stats['scale']:>7.2f} {stats['noise_scale']:>7.4f}")
        print(f"{'='*80}\n")


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
