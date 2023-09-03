"""Config for Go1 speed tracking environment."""
from ml_collections import ConfigDict
import numpy as np
import torch


def get_config():
  config = ConfigDict()

  gait_config = ConfigDict()
  gait_config.stepping_frequency = 1
  gait_config.initial_offset = np.array([0.05, 0.05, -0.4, -0.4],
                                        dtype=np.float32) * (2 * np.pi)
  gait_config.swing_ratio = np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32)
  config.gait = gait_config

  config.goal_lb = torch.tensor([1., 0.],
                                dtype=torch.float)  # Lin_x, Lin_y, Rot_z
  config.goal_ub = torch.tensor([1., 0.], dtype=torch.float)

  # Action: step_freq, height, vx, vy,  vz, roll, pitch, pitch_rate, yaw_rate
  config.include_gait_action = True
  config.include_foot_action = True
  config.mirror_foot_action = True

  # Fixed v-action
  config.action_lb = np.array([0.5] + [-0.3, -0.2, -2.4] * 2)
  config.action_ub = np.array([3.999] + [0.3, 2., -1.2] * 2)

  config.episode_length_s = 20.
  config.max_jumps = 10.
  config.env_dt = 0.01
  config.motor_strength_ratios = 1.  #np.array([1., 0.7, 0.7] * 4)  #(0.7, 1)  # 0.7
  config.motor_torque_delay_steps = 5
  config.use_yaw_feedback = False
  config.foot_friction = 1.  #0.7
  config.terminate_on_body_contact = True
  config.terminate_on_limb_contact = False
  config.terminate_on_height = 0.15
  config.use_penetrating_contact = False

  config.rewards = [
      ('alive', 0.02),
      ('upright', 0.02),
      ('contact_consistency', 0.008),
      ('energy_consumption', 1e-6),
      ('foot_slipping', 0.032),
      ('foot_clearance', 0.008),
      ('out_of_bound_action', 0.01),
      ('knee_contact', 0.064),
      ('stepping_freq', 0.008),
      ('com_distance_to_goal_squared', 0.016),
      ('com_height', 0.01),
  ]
  config.clip_negative_reward = False
  config.normalize_reward_by_phase = True

  config.terminal_rewards = []
  config.clip_negative_terminal_reward = False
  return config
