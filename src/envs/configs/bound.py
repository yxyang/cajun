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

  config.goal_lb = torch.tensor([0.3, 0.],
                                dtype=torch.float)  # Lin_x, Lin_y, Rot_z
  config.goal_ub = torch.tensor([1., 0.], dtype=torch.float)

  # Action: step_freq, height, vx, vy,  vz, roll, pitch, pitch_rate, yaw_rate
  config.include_gait_action = True
  config.include_foot_action = True
  config.mirror_foot_action = True

  # Fully Flexible
  config.action_lb = np.array(
      [0.5, -0.001, -3, -0.001, -3., -0.001, -0.001, -2.5, -0.001] +
      [-0.1, -0.0001, -0.001] * 2)
  config.action_ub = np.array(
      [3.999, 0.001, 3, 0.001, 3., 0.001, 0.001, 2.5, 0.001] +
      [0.1, 0.0001, 0.2] * 2)

  config.episode_length_s = 20.
  config.max_jumps = 10.
  config.env_dt = 0.01
  config.motor_strength_ratios = 1.
  config.motor_torque_delay_steps = 5
  config.use_yaw_feedback = False
  config.foot_friction = 1.  #0.7
  config.base_position_kp = np.array([0., 0., 0.])
  config.base_position_kd = np.array([10., 10., 10.])
  config.base_orientation_kp = np.array([50., 0., 0.])
  config.base_orientation_kd = np.array([10., 10., 10.])
  config.qp_foot_friction_coef = 0.6
  config.qp_weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
  config.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 1.5
  config.clip_grf_in_sim = True
  config.use_full_qp = False
  config.swing_foot_height = 0.
  config.swing_foot_landing_clearance = 0.
  config.terminate_on_body_contact = True
  config.terminate_on_limb_contact = False
  config.terminate_on_height = 0.15
  config.use_penetrating_contact = False

  config.rewards = [
      ('upright', 0.02),
      ('contact_consistency', 0.008),
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
