"""Solves the centroidal QP to compute desired foot torques."""
import time

from isaacgym.torch_utils import quat_mul, quat_from_euler_xyz, to_torch, quat_rotate
import numpy as np
import torch
from qpth.qp import QPFunction, QPSolvers

from src.robots.motors import MotorCommand
from src.utilities.rotation_utils import quat_to_rot_mat


@torch.jit.script
def quaternion_to_axis_angle(q):
  angle = 2 * torch.acos(torch.clip(q[:, 3], -0.99999, 0.99999))[:, None]
  norm = torch.clip(torch.linalg.norm(q[:, :3], dim=1), 1e-5, 1)[:, None]
  axis = q[:, :3] / norm
  return axis, angle


@torch.jit.script
def compute_orientation_error(desired_orientation_rpy,
                              base_orientation_quat,
                              device: str = 'cuda'):
  desired_quat = quat_from_euler_xyz(
      desired_orientation_rpy[:, 0], desired_orientation_rpy[:, 1],
      torch.zeros_like(desired_orientation_rpy[:, 2]))
  base_quat_inv = torch.clone(base_orientation_quat)
  base_quat_inv[:, -1] *= -1
  error_quat = quat_mul(desired_quat, base_quat_inv)
  axis, angle = quaternion_to_axis_angle(error_quat)
  angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)
  error_so3 = axis * angle
  return quat_rotate(base_orientation_quat, error_so3)


@torch.jit.script
def compute_desired_acc(
    base_orientation_rpy: torch.Tensor,
    base_position: torch.Tensor,
    base_angular_velocity_body_frame: torch.Tensor,
    base_velocity_body_frame: torch.Tensor,
    desired_base_orientation_rpy: torch.Tensor,
    desired_base_position: torch.Tensor,
    desired_angular_velocity: torch.Tensor,
    desired_linear_velocity: torch.Tensor,
    desired_angular_acceleration: torch.Tensor,
    desired_linear_acceleration: torch.Tensor,
    base_position_kp: torch.Tensor,
    base_position_kd: torch.Tensor,
    base_orientation_kp: torch.Tensor,
    base_orientation_kd: torch.Tensor,
    device: str = "cuda",
):
  base_rpy = base_orientation_rpy
  base_quat = quat_from_euler_xyz(
      base_rpy[:, 0], base_rpy[:, 1],
      torch.zeros_like(base_rpy[:, 0], device=device))
  base_rot_mat = quat_to_rot_mat(base_quat)
  base_rot_mat_t = torch.transpose(base_rot_mat, 1, 2)

  lin_pos_error = desired_base_position - base_position
  lin_pos_error[:, :2] = 0
  lin_vel_error = desired_linear_velocity - torch.matmul(
      base_rot_mat, base_velocity_body_frame[:, :, None])[:, :, 0]
  desired_lin_acc_gravity_frame = (base_position_kp * lin_pos_error +
                                   base_position_kd * lin_vel_error +
                                   desired_linear_acceleration)

  ang_pos_error = compute_orientation_error(desired_base_orientation_rpy,
                                            base_quat,
                                            device=device)
  ang_vel_error = desired_angular_velocity - torch.matmul(
      base_rot_mat, base_angular_velocity_body_frame[:, :, None])[:, :, 0]
  desired_ang_acc_gravity_frame = (base_orientation_kp * ang_pos_error +
                                   base_orientation_kd * ang_vel_error +
                                   desired_angular_acceleration)

  desired_lin_acc_body_frame = torch.matmul(
      base_rot_mat_t, desired_lin_acc_gravity_frame[:, :, None])[:, :, 0]
  desired_ang_acc_body_frame = torch.matmul(
      base_rot_mat_t, desired_ang_acc_gravity_frame[:, :, None])[:, :, 0]
  # print(f"Desired position: {desired_base_position}")
  # print(f"Current position: {base_position}")
  # print(f"Desired lin acc body: {desired_lin_acc_body_frame}")
  # print(f"Desired ang acc body: {desired_ang_acc_body_frame}")
  # ans = input("Any Key...")
  # if ans in ["y", "Y"]:
  #   import pdb
  #   pdb.set_trace()
  return torch.concatenate(
      (desired_lin_acc_body_frame, desired_ang_acc_body_frame), dim=1)


@torch.jit.script
def convert_to_skew_symmetric_batch(foot_positions):
  """
  Converts foot positions (nx4x3) into skew-symmetric ones (nx3x12)
  """
  n = foot_positions.shape[0]
  x = foot_positions[:, :, 0]
  y = foot_positions[:, :, 1]
  z = foot_positions[:, :, 2]
  zero = torch.zeros_like(x)
  skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).reshape(
      (n, 3, 3, 4))
  return torch.concatenate(
      [skew[:, :, :, 0], skew[:, :, :, 1], skew[:, :, :, 2], skew[:, :, :, 3]],
      dim=2)


@torch.jit.script
def construct_mass_mat(foot_positions,
                       foot_contact_state,
                       inv_mass,
                       inv_inertia,
                       device: str = 'cuda',
                       mask_noncontact_legs: bool = True):
  num_envs = foot_positions.shape[0]
  mass_mat = torch.zeros((num_envs, 6, 12), device=device)
  # Construct mass matrix
  inv_mass_concat = torch.concatenate([inv_mass] * 4, dim=1)
  mass_mat[:, :3] = inv_mass_concat[None, :, :]
  px = convert_to_skew_symmetric_batch(foot_positions)
  mass_mat[:, 3:6] = torch.matmul(inv_inertia, px)
  # Mark out non-contact legs
  if mask_noncontact_legs:
    non_contact_indices = torch.nonzero(torch.logical_not(foot_contact_state))
    env_id, leg_id = non_contact_indices[:, 0], non_contact_indices[:, 1]
    mass_mat[env_id, :, leg_id * 3] = 0
    mass_mat[env_id, :, leg_id * 3 + 1] = 0
    mass_mat[env_id, :, leg_id * 3 + 2] = 0
  return mass_mat


@torch.jit.script
def solve_grf(mass_mat,
              desired_acc,
              base_rot_mat_t,
              Wq,
              Wf: float,
              foot_friction_coef: float,
              clip_grf: bool,
              foot_contact_state,
              device: str = 'cuda'):
  num_envs = mass_mat.shape[0]
  g = torch.zeros((num_envs, 6), device=device)
  g[:, 2] = 9.8

  g[:, :3] = torch.matmul(base_rot_mat_t, g[:, :3, None])[:, :, 0]
  Q = torch.zeros((num_envs, 6, 6), device=device) + Wq[None, :]
  Wf_mat = torch.eye(12, device=device) * Wf
  R = torch.zeros((num_envs, 12, 12), device=device) + Wf_mat[None, :]

  quad_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                        mass_mat) + R
  linear_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                          (g + desired_acc)[:, :, None])[:, :, 0]
  grf = torch.linalg.solve(quad_term, linear_term)

  base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)
  grf = grf.reshape((-1, 4, 3))
  grf_world = torch.transpose(
      torch.bmm(base_rot_mat, torch.transpose(grf, 1, 2)), 1, 2)
  if clip_grf:
    grf_world[:, :, 2] = grf_world[:, :, 2].clip(min=10, max=130)
    grf_world[:, :, 2] *= foot_contact_state
  friction_force = torch.norm(grf_world[:, :, :2], dim=2) + 0.001
  max_friction_force = foot_friction_coef * grf_world[:, :, 2].clip(min=0)
  multiplier = torch.where(friction_force < max_friction_force, 1,
                           max_friction_force / friction_force)
  if clip_grf:
    grf_world[:, :, :2] *= multiplier[:, :, None]
  grf = torch.transpose(
      torch.bmm(base_rot_mat_t, torch.transpose(grf_world, 1, 2)), 1, 2)
  grf = grf.reshape((-1, 12))

  # Convert to motor torques
  solved_acc = torch.bmm(mass_mat, grf[:, :, None])[:, :, 0] - g
  qp_cost = torch.bmm(
      torch.bmm((solved_acc - desired_acc)[:, :, None].transpose(1, 2), Q),
      (solved_acc - desired_acc)[:, :, None])[:, 0, 0]
  return grf, solved_acc, qp_cost, torch.sum(
      friction_force > max_friction_force + 1, dim=1)


# @torch.jit.script
def solve_grf_qpth(mass_mat,
                   desired_acc,
                   base_rot_mat_t,
                   Wq,
                   Wf: float,
                   foot_friction_coef: float,
                   clip_grf: bool,
                   foot_contact_state,
                   device: str = 'cuda'):
  num_envs = mass_mat.shape[0]
  g = torch.zeros((num_envs, 6), device=device)
  g[:, 2] = 9.8

  g[:, :3] = torch.matmul(base_rot_mat_t, g[:, :3, None])[:, :, 0]
  Q = torch.zeros((num_envs, 6, 6), device=device) + Wq[None, :]
  Wf_mat = torch.eye(12, device=device) * Wf
  R = torch.zeros((num_envs, 12, 12), device=device) + Wf_mat[None, :]

  quad_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                        mass_mat) + R
  linear_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                          (g + desired_acc)[:, :, None])[:, :, 0]

  G = torch.zeros((mass_mat.shape[0], 24, 12), device=device)
  h = torch.zeros((mass_mat.shape[0], 24), device=device) + 1e-3
  base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)
  for leg_id in range(4):
    G[:, leg_id * 2, leg_id * 3 + 2] = 1
    G[:, leg_id * 2 + 1, leg_id * 3 + 2] = -1

    row_id, col_id = 8 + leg_id * 4, leg_id * 3
    G[:, row_id, col_id] = 1
    G[:, row_id, col_id + 2] = -foot_friction_coef

    G[:, row_id + 1, col_id] = -1
    G[:, row_id + 1, col_id + 2] = -foot_friction_coef

    G[:, row_id + 2, col_id + 1] = 1
    G[:, row_id + 2, col_id + 2] = -foot_friction_coef

    G[:, row_id + 3, col_id + 1] = -1
    G[:, row_id + 3, col_id + 2] = -foot_friction_coef
    G[:, row_id:row_id + 4, col_id:col_id + 3] = torch.bmm(
        G[:, row_id:row_id + 4, col_id:col_id + 3], base_rot_mat)

  contact_ids = foot_contact_state.nonzero()

  h[contact_ids[:, 0], contact_ids[:, 1] * 2] = 130
  h[contact_ids[:, 0], contact_ids[:, 1] * 2 + 1] = -10
  e = torch.autograd.Variable(torch.Tensor())

  qf = QPFunction(verbose=-1,
                  check_Q_spd=False,
                  eps=1e-3,
                  solver=QPSolvers.PDIPM_BATCHED)
  grf = qf(quad_term.double(), -linear_term.double(), G.double(), h.double(),
           e, e).float()
  # print(grf)
  # ans = input("Any Key...")
  # if ans in ["Y", "y"]:
  #   import pdb
  #   pdb.set_trace()
  solved_acc = torch.bmm(mass_mat, grf[:, :, None])[:, :, 0] - g
  qp_cost = torch.bmm(
      torch.bmm((solved_acc - desired_acc)[:, :, None].transpose(1, 2), Q),
      (solved_acc - desired_acc)[:, :, None])[:, 0, 0]

  return grf, solved_acc, qp_cost, torch.zeros(mass_mat.shape[0],
                                               device=device)


class QPTorqueOptimizer:
  """Centroidal QP controller to optimize for joint torques."""
  def __init__(self,
               robot,
               base_position_kp=np.array([0., 0., 50]),
               base_position_kd=np.array([10., 10., 10.]),
               base_orientation_kp=np.array([50., 50., 0.]),
               base_orientation_kd=np.array([10., 10., 10.]),
               weight_ddq=np.diag([1., 1., 10., 10., 10., 1.]),
               weight_grf=1e-4,
               body_mass=13.076,
               body_inertia=np.array([0.14, 0.35, 0.35]) * 0.5,
               desired_body_height=0.26,
               foot_friction_coef=0.7,
               clip_grf=False,
               use_full_qp=False):
    """Initializes the controller with desired weights and gains."""
    self._robot = robot
    self._device = self._robot._device
    self._num_envs = self._robot.num_envs
    self._clip_grf = clip_grf
    self._use_full_qp = use_full_qp

    self._base_orientation_kp = to_torch(base_orientation_kp,
                                         device=self._device)
    self._base_orientation_kp = torch.stack([self._base_orientation_kp] *
                                            self._num_envs,
                                            dim=0)
    self._base_orientation_kd = to_torch(base_orientation_kd,
                                         device=self._device)
    self._base_orientation_kd = torch.stack([self._base_orientation_kd] *
                                            self._num_envs,
                                            dim=0)
    self._base_position_kp = to_torch(base_position_kp, device=self._device)
    self._base_position_kp = torch.stack([self._base_position_kp] *
                                         self._num_envs,
                                         dim=0)
    self._base_position_kd = to_torch(base_position_kd, device=self._device)
    self._base_position_kd = torch.stack([self._base_position_kd] *
                                         self._num_envs,
                                         dim=0)
    self._desired_base_orientation_rpy = torch.zeros((self._num_envs, 3),
                                                     device=self._device)
    self._desired_base_position = torch.zeros((self._num_envs, 3),
                                              device=self._device)
    self._desired_base_position[:, 2] = desired_body_height
    self._desired_linear_velocity = torch.zeros((self._num_envs, 3),
                                                device=self._device)
    self._desired_angular_velocity = torch.zeros((self._num_envs, 3),
                                                 device=self._device)
    self._desired_linear_acceleration = torch.zeros((self._num_envs, 3),
                                                    device=self._device)
    self._desired_angular_acceleration = torch.zeros((self._num_envs, 3),
                                                     device=self._device)
    self._Wq = to_torch(weight_ddq, device=self._device, dtype=torch.float32)
    self._Wf = to_torch(weight_grf, device=self._device)
    self._foot_friction_coef = foot_friction_coef
    self._inv_mass = torch.eye(3, device=self._device) / body_mass
    self._inv_inertia = torch.linalg.inv(
        torch.diag(to_torch(body_inertia, device=self._device)))

  def _solve_joint_torques(self, foot_contact_state, desired_com_ddq):
    """Solves centroidal QP to find desired joint torques."""
    self._mass_mat = construct_mass_mat(
        self._robot.foot_positions_in_base_frame,
        foot_contact_state,
        self._inv_mass,
        self._inv_inertia,
        mask_noncontact_legs=not self._use_full_qp,
        device=self._device)

    # Solve QP
    if self._use_full_qp:
      grf, solved_acc, qp_cost, num_clips = solve_grf_qpth(
          self._mass_mat,
          desired_com_ddq,
          self._robot.base_rot_mat_t,
          self._Wq,
          self._Wf,
          self._foot_friction_coef,
          self._clip_grf,
          foot_contact_state,
          device=self._device)
    else:
      grf, solved_acc, qp_cost, num_clips = solve_grf(
          self._mass_mat,
          desired_com_ddq,
          self._robot.base_rot_mat_t,
          self._Wq,
          self._Wf,
          self._foot_friction_coef,
          self._clip_grf,
          foot_contact_state,
          device=self._device)

    all_foot_jacobian = self._robot.all_foot_jacobian
    motor_torques = -torch.bmm(grf[:, None, :], all_foot_jacobian)[:, 0]
    return motor_torques, solved_acc, grf, qp_cost, num_clips

  def compute_joint_command(self, foot_contact_state: torch.Tensor,
                            desired_base_orientation_rpy: torch.Tensor,
                            desired_base_position: torch.Tensor,
                            desired_foot_position: torch.Tensor,
                            desired_angular_velocity: torch.Tensor,
                            desired_linear_velocity: torch.Tensor,
                            desired_foot_velocity: torch.Tensor,
                            desired_angular_acceleration: torch.Tensor,
                            desired_linear_acceleration: torch.Tensor,
                            desired_foot_acceleration: torch.Tensor):
    desired_acc_body_frame = compute_desired_acc(
        self._robot.base_orientation_rpy,
        self._robot.base_position,
        self._robot.base_angular_velocity_body_frame,
        self._robot.base_velocity_body_frame,
        desired_base_orientation_rpy,
        desired_base_position,
        desired_angular_velocity,
        desired_linear_velocity,
        desired_angular_acceleration,
        desired_linear_acceleration,
        self._base_position_kp,
        self._base_position_kd,
        self._base_orientation_kp,
        self._base_orientation_kd,
        device=self._device)
    desired_acc_body_frame = torch.clip(
        desired_acc_body_frame,
        to_torch([-30, -30, -10, -20, -20, -20], device=self._device),
        to_torch([30, 30, 30, 20, 20, 20], device=self._device))
    motor_torques, solved_acc, grf, qp_cost, num_clips = self._solve_joint_torques(
        foot_contact_state, desired_acc_body_frame)
    foot_position_local = torch.bmm(self._robot.base_rot_mat_t,
                                    desired_foot_position.transpose(
                                        1, 2)).transpose(1, 2)
    foot_position_local[:, :, 2] = torch.clip(foot_position_local[:, :, 2],
                                              min=-0.35,
                                              max=-0.1)

    desired_motor_position = self._robot.get_motor_angles_from_foot_positions(
        foot_position_local)

    contact_state_expanded = foot_contact_state.repeat_interleave(3, dim=1)

    desired_position = torch.where(contact_state_expanded,
                                   self._robot.motor_positions,
                                   desired_motor_position)
    desired_velocity = torch.where(contact_state_expanded,
                                   self._robot.motor_velocities,
                                   torch.zeros_like(motor_torques))
    desired_torque = torch.where(contact_state_expanded, motor_torques,
                                 torch.zeros_like(motor_torques))
    desired_torque = torch.clip(desired_torque,
                                max=self._robot.motor_group.max_torques,
                                min=self._robot.motor_group.min_torques)
    # print(self._robot.time_since_reset)
    # print("Contact: {}".format(foot_contact_state))
    # print("Desired pos: {}".format(desired_base_position))
    # print("Current vel: {}".format(self._robot.base_velocity_body_frame))
    # print("Desired vel: {}".format(desired_linear_velocity))
    # print(f"GRF: {grf.reshape((4, 3))}")
    # print("Desired acc: {}".format(desired_acc_body_frame))
    # print("Solved acc: {}".format(solved_acc))
    # print(f"Desired torque: {desired_torque}")
    # ans = input("Any Key...")
    # if ans in ['y', 'Y']:
    #   import pdb
    #   pdb.set_trace()
    return MotorCommand(
        desired_position=desired_position,
        kp=torch.ones_like(self._robot.motor_group.kps) * 30,
        desired_velocity=desired_velocity,
        kd=torch.ones_like(self._robot.motor_group.kds) * 1,
        desired_extra_torque=desired_torque
    ), desired_acc_body_frame, solved_acc, qp_cost, num_clips

  def get_action(self, foot_contact_state: torch.Tensor,
                 swing_foot_position: torch.Tensor):
    """Computes motor actions."""
    return self.compute_joint_command(
        foot_contact_state=foot_contact_state,
        desired_base_orientation_rpy=self._desired_base_orientation_rpy,
        desired_base_position=self._desired_base_position,
        desired_foot_position=swing_foot_position,
        desired_angular_velocity=self._desired_angular_velocity,
        desired_linear_velocity=self._desired_linear_velocity,
        desired_foot_velocity=torch.zeros(12),
        desired_angular_acceleration=self._desired_angular_acceleration,
        desired_linear_acceleration=self._desired_linear_acceleration,
        desired_foot_acceleration=torch.zeros(12))

  def get_action_with_acc(
      self,
      foot_contact_state: torch.Tensor,
      desired_acc_body_frame: torch.Tensor,
      desired_foot_position: torch.Tensor,
  ):
    motor_torques, solved_acc, grf, qp_cost, num_clips = self._solve_joint_torques(
        foot_contact_state, desired_acc_body_frame)
    foot_position_local = torch.bmm(self._robot.base_rot_mat_t,
                                    desired_foot_position.transpose(
                                        1, 2)).transpose(1, 2)
    foot_position_local[:, :, 2] = torch.clip(foot_position_local[:, :, 2],
                                              min=-0.35,
                                              max=-0.1)

    desired_motor_position = self._robot.get_motor_angles_from_foot_positions(
        foot_position_local)

    contact_state_expanded = foot_contact_state.repeat_interleave(3, dim=1)

    desired_position = torch.where(contact_state_expanded,
                                   self._robot.motor_positions,
                                   desired_motor_position)
    desired_velocity = torch.where(contact_state_expanded,
                                   self._robot.motor_velocities,
                                   torch.zeros_like(motor_torques))
    desired_torque = torch.where(contact_state_expanded, motor_torques,
                                 torch.zeros_like(motor_torques))
    desired_torque = torch.clip(desired_torque,
                                max=self._robot.motor_group.max_torques,
                                min=self._robot.motor_group.min_torques)
    return MotorCommand(
        desired_position=desired_position,
        kp=torch.ones_like(self._robot.motor_group.kps) * 30,
        desired_velocity=desired_velocity,
        kd=torch.ones_like(self._robot.motor_group.kds) * 1,
        desired_extra_torque=desired_torque
    ), desired_acc_body_frame, solved_acc, qp_cost, num_clips

  @property
  def desired_base_position(self) -> torch.Tensor:
    return self._desired_base_position

  @desired_base_position.setter
  def desired_base_position(self, base_position: float):
    self._desired_base_position = to_torch(base_position, device=self._device)

  @property
  def desired_base_orientation_rpy(self) -> torch.Tensor:
    return self._desired_base_orientation_rpy

  @desired_base_orientation_rpy.setter
  def desired_base_orientation_rpy(self, orientation_rpy: torch.Tensor):
    self._desired_base_orientation_rpy = to_torch(orientation_rpy,
                                                  device=self._device)

  @property
  def desired_linear_velocity(self) -> torch.Tensor:
    return self._desired_linear_velocity

  @desired_linear_velocity.setter
  def desired_linear_velocity(self, desired_linear_velocity: torch.Tensor):
    self._desired_linear_velocity = to_torch(desired_linear_velocity,
                                             device=self._device)

  @property
  def desired_angular_velocity(self) -> torch.Tensor:
    return self._desired_angular_velocity

  @desired_angular_velocity.setter
  def desired_angular_velocity(self, desired_angular_velocity: torch.Tensor):
    self._desired_angular_velocity = to_torch(desired_angular_velocity,
                                              device=self._device)

  @property
  def desired_linear_acceleration(self):
    return self._desired_linear_acceleration

  @desired_linear_acceleration.setter
  def desired_linear_acceleration(self,
                                  desired_linear_acceleration: torch.Tensor):
    self._desired_linear_acceleration = to_torch(desired_linear_acceleration,
                                                 device=self._device)

  @property
  def desired_angular_acceleration(self):
    return self._desired_angular_acceleration

  @desired_angular_acceleration.setter
  def desired_angular_acceleration(self,
                                   desired_angular_acceleration: torch.Tensor):
    self._desired_angular_acceleration = to_torch(desired_angular_acceleration,
                                                  device=self._device)
