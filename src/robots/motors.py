"""Implements motor models for different motor control modes."""
from collections import deque

from dataclasses import dataclass
import enum
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from isaacgym.torch_utils import to_torch
import torch

_ARRAY = Sequence[float]
_FloatOrArray = Union[float, _ARRAY]


class MotorControlMode(enum.Enum):
  POSITION = 0
  HYBRID = 1


@dataclass
class MotorCommand:
  desired_position: torch.Tensor = torch.zeros(12)
  kp: torch.Tensor = torch.zeros(12)
  desired_velocity: torch.Tensor = torch.zeros(12)
  kd: torch.Tensor = torch.zeros(12)
  desired_extra_torque: torch.Tensor = torch.zeros(12)


class MotorModel:
  """Implements a simple DC motor model for simulation.
    To accurately model the motor behaviors, the `MotorGroup` class converts
    all motor commands into torques, which is sent directly to the simulator.
    Each `MotorModel` describes a characteristics of a particular motor.
    NOTE: Until functionality is added to MotorModel, it is effectively
    equivalent to a `dataclass`.
    """

  # TODO(yxyang): Complete documentation of motors with description of units
  # (e.g. rads/s etc.)

  def __init__(
      self,
      name: Optional[str] = None,
      motor_control_mode: MotorControlMode = MotorControlMode.POSITION,
      init_position: float = 0.0,
      min_position: float = 0.0,
      max_position: float = 0.0,
      min_velocity: float = 0.0,
      max_velocity: float = 0.0,
      min_torque: float = 0.0,
      max_torque: float = 0.0,
      kp: float = 0.0,
      kd: float = 0.0,
  ) -> None:

    self._name = name
    self._motor_control_mode = motor_control_mode
    self._init_position = init_position
    self._min_position = min_position
    self._max_position = max_position
    self._min_velocity = min_velocity
    self._max_velocity = max_velocity
    self._min_torque = min_torque
    self._max_torque = max_torque
    self._kp = kp
    self._kd = kd


class MotorGroup:
  """Models the behavior of a group of motors."""

  def __init__(self,
               device: str,
               num_envs: int,
               motors: Tuple[MotorModel, ...] = (),
               torque_delay_steps: int = 0):

    self._motors = motors
    self._num_envs = num_envs
    self._num_motors = len(motors)
    self._motor_control_mode = motors[0]._motor_control_mode
    self._device = device
    self._strength_ratios = torch.ones((self._num_envs, self._num_motors),
                                       device=device)
    self._init_positions = to_torch([motor._init_position for motor in motors],
                                    device=device)
    self._min_positions = to_torch([motor._min_position for motor in motors],
                                   device=device)
    self._max_positions = to_torch([motor._max_position for motor in motors],
                                   device=device)
    self._min_velocities = to_torch([motor._min_velocity for motor in motors],
                                    device=device)
    self._max_velocities = to_torch([motor._max_velocity for motor in motors],
                                    device=device)
    self._min_torques = to_torch([motor._min_torque for motor in motors],
                                 device=device)
    self._max_torques = to_torch([motor._max_torque for motor in motors],
                                 device=device)
    self._kps = to_torch([motor._kp for motor in motors], device=device)
    self._kds = to_torch([motor._kd for motor in motors], device=device)
    self._torque_history = deque(maxlen=torque_delay_steps + 1)
    self._torque_history.append(
        torch.zeros((self._num_envs, self._num_motors), device=self._device))
    self._torque_output = torch.zeros((self._num_envs, self._num_motors),
                                      device=self._device)
    self._true_motor_torque = torch.zeros((self._num_envs, self._num_motors),
                                          device=self._device)

  def _clip_torques(self, desired_torque: _ARRAY,
                    current_motor_velocity: _ARRAY):
    torque_ub = torch.where(
        current_motor_velocity < 0, self._max_torques,
        self._max_torques * (1 - current_motor_velocity / self._max_velocities))
    torque_lb = torch.where(
        current_motor_velocity < 0,
        self._min_torques * (1 - current_motor_velocity / self._min_velocities),
        self._min_torques)

    return torch.clip(desired_torque, torque_lb, torque_ub)

  def convert_to_torque(
      self,
      command: MotorCommand,
      current_position: _ARRAY,
      current_velocity: _ARRAY,
      motor_control_mode: Optional[MotorControlMode] = None,
  ):
    """Converts the given motor command into motor torques."""
    motor_control_mode = motor_control_mode or self._motor_control_mode
    if motor_control_mode == MotorControlMode.POSITION:
      desired_position = command.desired_position
      kp = self._kps
      desired_velocity = torch.zeros((self._num_envs, self._num_motors),
                                     device=self._device)
      kd = self._kds
      # desired_torque = torch.zeros((self._num_envs, self._num_motors),
      #                              device=self._device)
    else:  # HYBRID case
      desired_position = command.desired_position
      kp = command.kp
      desired_velocity = command.desired_velocity
      kd = command.kd
      self._torque_history.append(command.desired_extra_torque)
      self._torque_output = 0 * self._torque_output + 1. * self._torque_history[
          0]

    total_torque = (kp * (desired_position - current_position) + kd *
                    (desired_velocity - current_velocity) + self._torque_output)
    applied_torque = self._clip_torques(total_torque, current_velocity)
    applied_torque *= self._strength_ratios

    return applied_torque, total_torque

  @property
  def motor_control_mode(self):
    return self._motor_control_mode

  @property
  def kps(self):
    return self._kps

  @kps.setter
  def kps(self, value: _FloatOrArray):
    self._kps = torch.ones(self._num_motors) * value

  @property
  def kds(self):
    return self._kds

  @kds.setter
  def kds(self, value: _FloatOrArray):
    self._kds = torch.ones(self._num_motors) * value

  @property
  def strength_ratios(self):
    return self._strength_ratios

  @strength_ratios.setter
  def strength_ratios(self, value: _FloatOrArray):
    self._strength_ratios = torch.ones(
        self._num_motors, device=self._device) * to_torch(value,
                                                          device=self._device)

  @property
  def init_positions(self):
    return self._init_positions

  @init_positions.setter
  def init_positions(self, value: _ARRAY):
    self._init_positions = value

  @property
  def num_motors(self):
    return self._num_motors

  @property
  def min_positions(self):
    return self._min_positions

  @property
  def max_positions(self):
    return self._max_positions

  @property
  def min_torques(self):
    return self._min_torques

  @property
  def max_torques(self):
    return self._max_torques
