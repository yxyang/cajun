"""General class for (vectorized) robots."""
import os
import sys
from typing import Any, List

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.configs.defaults import asset_options as asset_options_config
from src.utilities.rotation_utils import quat_to_rot_mat, get_euler_xyz_from_quaternion


def angle_normalize(x):
  return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


class Robot:
  """General class for simulated quadrupedal robot."""
  def __init__(
      self,
      sim: Any,
      viewer: Any,
      num_envs: int,
      init_positions: torch.Tensor,
      urdf_path: str,
      sim_config: ml_collections.ConfigDict,
      motors: Any,
      feet_names: List[str],
      calf_names: List[str],
      thigh_names: List[str],
  ):
    """Initializes the robot class."""
    self._gym = gymapi.acquire_gym()
    self._sim = sim
    self._viewer = viewer
    self._enable_viewer_sync = True
    self._sim_config = sim_config
    self._device = self._sim_config.sim_device
    self._num_envs = num_envs
    self._motors = motors
    self._feet_names = feet_names
    self._calf_names = calf_names
    self._thigh_names = thigh_names

    self._base_init_state = self._compute_base_init_state(init_positions)
    self._init_motor_angles = self._motors.init_positions
    self._envs = []
    self._actors = []
    self._time_since_reset = torch.zeros(self._num_envs, device=self._device)

    if "cuda" in self._device:
      torch._C._jit_set_profiling_mode(False)
      torch._C._jit_set_profiling_executor(False)

    self._load_urdf(urdf_path)
    self._gym.prepare_sim(self._sim)
    self._init_buffers()

    self._post_physics_step()
    # self.reset()

  def _compute_base_init_state(self, init_positions: torch.Tensor):
    """Computes desired init state for CoM (position and velocity)."""
    num_envs = init_positions.shape[0]
    init_state_list = [0., 0., 0.] + [0., 0., 0., 1.] + [0., 0., 0.
                                                         ] + [0., 0., 0.]
    # init_state_list = [0., 0., 0.] + [0., 0., 0.7071, 0.7071] + [0., 0., 0.
    #                                                      ] + [0., 0., 0.]
    # init_state_list = [0., 0., 0.] + [ 0.0499792, 0, 0, 0.9987503
    #                                       ] + [0., 0., 0.] + [0., 0., 0.]
    init_states = np.stack([init_state_list] * num_envs, axis=0)
    init_states = to_torch(init_states, device=self._device)
    init_states[:, :3] = init_positions
    return to_torch(init_states, device=self._device)

  def _load_urdf(self, urdf_path):
    asset_root = os.path.dirname(urdf_path)
    asset_file = os.path.basename(urdf_path)
    asset_config = asset_options_config.get_config()
    self._robot_asset = self._gym.load_asset(self._sim, asset_root, asset_file,
                                             asset_config.asset_options)
    self._num_dof = self._gym.get_asset_dof_count(self._robot_asset)
    self._num_bodies = self._gym.get_asset_rigid_body_count(self._robot_asset)

    env_lower = gymapi.Vec3(0., 0., 0.)
    env_upper = gymapi.Vec3(0., 0., 0.)
    for i in range(self._num_envs):
      env_handle = self._gym.create_env(self._sim, env_lower, env_upper,
                                        int(np.sqrt(self._num_envs)))
      start_pose = gymapi.Transform()
      start_pose.p = gymapi.Vec3(*self._base_init_state[i, :3])
      actor_handle = self._gym.create_actor(env_handle, self._robot_asset,
                                            start_pose, "actor", i,
                                            asset_config.self_collisions, 0)
      self._gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
      self._envs.append(env_handle)
      self._actors.append(actor_handle)

    self._feet_indices = torch.zeros(len(self._feet_names),
                                     dtype=torch.long,
                                     device=self._device,
                                     requires_grad=False)
    for i in range(len(self._feet_names)):
      self._feet_indices[i] = self._gym.find_actor_rigid_body_handle(
          self._envs[0], self._actors[0], self._feet_names[i])

    self._calf_indices = torch.zeros(len(self._calf_names),
                                     dtype=torch.long,
                                     device=self._device,
                                     requires_grad=False)
    for i in range(len(self._calf_names)):
      self._calf_indices[i] = self._gym.find_actor_rigid_body_handle(
          self._envs[0], self._actors[0], self._calf_names[i])

    self._thigh_indices = torch.zeros(len(self._thigh_names),
                                      dtype=torch.long,
                                      device=self._device,
                                      requires_grad=False)
    for i in range(len(self._thigh_names)):
      self._thigh_indices[i] = self._gym.find_actor_rigid_body_handle(
          self._envs[0], self._actors[0], self._thigh_names[i])

    self._body_indices = torch.zeros(self._num_bodies - len(self._feet_names) -
                                     len(self._thigh_names) -
                                     len(self._calf_names),
                                     dtype=torch.long,
                                     device=self._device)
    all_body_names = self._gym.get_actor_rigid_body_names(self._envs[0], 0)
    self._body_names = []
    limb_names = self._thigh_names + self._calf_names + self._feet_names
    idx = 0
    for name in all_body_names:
      if name not in limb_names:
        self._body_indices[idx] = self._gym.find_actor_rigid_body_handle(
            self._envs[0], self._actors[0], name)
        idx += 1
        self._body_names.append(name)

  def set_foot_friction(self, friction_coef, env_id=0):
    rigid_shape_props = self._gym.get_actor_rigid_shape_properties(
        self._envs[env_id], self._actors[env_id])
    for idx in range(len(rigid_shape_props)):
      rigid_shape_props[idx].friction = friction_coef
    self._gym.set_actor_rigid_shape_properties(self._envs[env_id],
                                               self._actors[env_id],
                                               rigid_shape_props)
    # import pdb
    # pdb.set_trace()

  def set_foot_frictions(self, friction_coefs, env_ids=None):
    if env_ids is None:
      env_ids = np.arange(self._num_envs)
    friction_coefs = friction_coefs * np.ones(self._num_envs)
    for env_id, friction_coef in zip(env_ids, friction_coefs):
      self.set_foot_friction(friction_coef, env_id=env_id)

  def _init_buffers(self):
    # get gym GPU state tensors
    actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
    dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
    net_contact_forces = self._gym.acquire_net_contact_force_tensor(self._sim)
    rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
    dof_force = self._gym.acquire_dof_force_tensor(self._sim)
    jacobians = self._gym.acquire_jacobian_tensor(self._sim, "actor")

    self._gym.refresh_dof_state_tensor(self._sim)
    self._gym.refresh_actor_root_state_tensor(self._sim)
    self._gym.refresh_net_contact_force_tensor(self._sim)

    # Robot state buffers
    self._root_states = gymtorch.wrap_tensor(actor_root_state)
    self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    self._rigid_body_state = gymtorch.wrap_tensor(
        rigid_body_state)[:self._num_envs * self._num_bodies, :]
    self._motor_positions = self._dof_state.view(self._num_envs, self._num_dof,
                                                 2)[..., 0]
    self._motor_velocities = self._dof_state.view(self._num_envs,
                                                  self._num_dof, 2)[..., 1]
    self._base_quat = self._root_states[:, 3:7]
    self._base_rot_mat = quat_to_rot_mat(self._base_quat)
    self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)

    self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
        self._num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
    self._motor_torques = gymtorch.wrap_tensor(dof_force).view(
        self._num_envs, self._num_dof)
    self._jacobian = gymtorch.wrap_tensor(jacobians)
    self._base_lin_vel_world = self._root_states[:, 7:10]
    self._base_ang_vel_world = self._root_states[:, 10:13]
    self._gravity_vec = torch.stack(
        [to_torch([0., 0., 1.], device=self._device)] * self._num_envs)
    self._projected_gravity = torch.bmm(self._base_rot_mat_t,
                                        self._gravity_vec[:, :, None])[:, :, 0]
    self._foot_velocities = self._rigid_body_state.view(
        self._num_envs, self._num_bodies, 13)[:, self._feet_indices, 7:10]
    self._foot_positions = self._rigid_body_state.view(self._num_envs,
                                                       self._num_bodies,
                                                       13)[:,
                                                           self._feet_indices,
                                                           0:3]
    # Other useful buffers
    self._torques = torch.zeros(self._num_envs,
                                self._num_dof,
                                dtype=torch.float,
                                device=self._device,
                                requires_grad=False)

  def reset(self):
    self.reset_idx(torch.arange(self._num_envs, device=self._device))

  def reset_idx(self, env_ids):
    if len(env_ids) == 0:
      return
    env_ids_int32 = env_ids.to(dtype=torch.int32)
    self._time_since_reset[env_ids] = 0

    # Reset root states:
    self._root_states[env_ids] = self._base_init_state[env_ids]

    self._gym.set_actor_root_state_tensor_indexed(
        self._sim, gymtorch.unwrap_tensor(self._root_states),
        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # Reset dofs
    self._motor_positions[env_ids] = to_torch(self._init_motor_angles,
                                              device=self._device,
                                              dtype=torch.float)
    self._motor_velocities[env_ids] = 0.

    self._gym.set_dof_state_tensor_indexed(
        self._sim, gymtorch.unwrap_tensor(self._dof_state),
        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    if len(env_ids) == self._num_envs:
      self._gym.simulate(self._sim)
    self._gym.refresh_dof_state_tensor(self._sim)
    self._post_physics_step()

  def step(self, action):
    for _ in range(self._sim_config.action_repeat):
      self._torques, _ = self.motor_group.convert_to_torque(
          action, self._motor_positions, self._motor_velocities)
      self._gym.set_dof_actuation_force_tensor(
          self._sim, gymtorch.unwrap_tensor(self._torques))
      self._gym.simulate(self._sim)
      if self._device == "cpu":
        self._gym.fetch_results(self._sim, True)
      self._gym.refresh_dof_state_tensor(self._sim)
      self._time_since_reset += self._sim_config.sim_params.dt

    self._post_physics_step()

  def _post_physics_step(self):
    self._gym.refresh_actor_root_state_tensor(self._sim)
    self._gym.refresh_net_contact_force_tensor(self._sim)
    self._gym.refresh_rigid_body_state_tensor(self._sim)
    self._gym.refresh_dof_force_tensor(self._sim)
    self._gym.refresh_jacobian_tensors(self._sim)
    self._base_quat[:] = self._root_states[:, 3:7]
    self._base_rot_mat = quat_to_rot_mat(self._base_quat)
    self._base_rot_mat_t = torch.transpose(self._base_rot_mat, 1, 2)
    self._base_lin_vel_world = self._root_states[:, 7:10]
    self._base_ang_vel_world = self._root_states[:, 10:13]
    self._projected_gravity[:] = torch.bmm(self._base_rot_mat_t,
                                           self._gravity_vec[:, :, None])[:, :,
                                                                          0]
    self._foot_velocities = self._rigid_body_state.view(
        self._num_envs, self._num_bodies, 13)[:, self._feet_indices, 7:10]
    self._foot_positions = self._rigid_body_state.view(self._num_envs,
                                                       self._num_bodies,
                                                       13)[:,
                                                           self._feet_indices,
                                                           0:3]

  def render(self, sync_frame_time=True):
    if self._viewer:
      # check for window closed
      if self._gym.query_viewer_has_closed(self._viewer):
        sys.exit()

      mean_pos = torch.min(self.base_position_world,
                           dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
      # mean_pos = torch.min(self.base_position_world,
      #                      dim=0)[0].cpu().numpy() + np.array([0.5, -1., 0.])
      target_pos = torch.mean(self.base_position_world,
                              dim=0).cpu().numpy() + np.array([0., 2., -0.5])
      cam_pos = gymapi.Vec3(*mean_pos)
      cam_target = gymapi.Vec3(*target_pos)
      self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)

      if self._device != "cpu":
        self._gym.fetch_results(self._sim, True)

      # step graphics
      self._gym.step_graphics(self._sim)
      self._gym.draw_viewer(self._viewer, self._sim, True)
      if sync_frame_time:
        self._gym.sync_frame_time(self._sim)

  def get_motor_angles_from_foot_positions(self, foot_local_positions):
    raise NotImplementedError()

  def update_init_positions(self, env_ids, init_positions):
    self._base_init_state[env_ids] = self._compute_base_init_state(
        init_positions)

  @property
  def base_position(self):
    base_position = torch.clone(self._root_states[:, :3])
    return base_position

  @property
  def base_position_world(self):
    return self._root_states[:, :3]

  @property
  def base_orientation_rpy(self):
    return angle_normalize(
        get_euler_xyz_from_quaternion(self._root_states[:, 3:7]))

  @property
  def base_orientation_quat(self):
    return self._root_states[:, 3:7]

  @property
  def projected_gravity(self):
    return self._projected_gravity

  @property
  def base_rot_mat(self):
    return self._base_rot_mat

  @property
  def base_rot_mat_t(self):
    return self._base_rot_mat_t

  @property
  def base_velocity_world_frame(self):
    return self._base_lin_vel_world

  @property
  def base_velocity_body_frame(self):
    return torch.bmm(self._base_rot_mat_t, self._root_states[:, 7:10,
                                                             None])[:, :, 0]

  @property
  def base_angular_velocity_world_frame(self):
    return self._base_ang_vel_world

  @property
  def base_angular_velocity_body_frame(self):
    return torch.bmm(self._base_rot_mat_t, self._root_states[:, 10:13,
                                                             None])[:, :, 0]

  @property
  def motor_positions(self):
    return torch.clone(self._motor_positions)

  @property
  def motor_velocities(self):
    return torch.clone(self._motor_velocities)

  @property
  def motor_torques(self):
    return torch.clone(self._torques)

  @property
  def foot_positions_in_base_frame(self):
    foot_positions_world_frame = self._foot_positions
    base_position_world_frame = self._root_states[:, :3]
    # num_env x 4 x 3
    foot_position = (foot_positions_world_frame -
                     base_position_world_frame[:, None, :])
    return torch.matmul(self._base_rot_mat_t,
                        foot_position.transpose(1, 2)).transpose(1, 2)

  @property
  def foot_positions_in_world_frame(self):
    return torch.clone(self._foot_positions)

  @property
  def foot_height(self):
    return self._foot_positions[:, :, 2]

  @property
  def foot_velocities_in_base_frame(self):
    foot_vels = torch.bmm(self.all_foot_jacobian,
                          self.motor_velocities[:, :, None]).squeeze()
    return foot_vels.reshape((self._num_envs, 4, 3))

  @property
  def foot_velocities_in_world_frame(self):
    return self._foot_velocities

  @property
  def foot_contacts(self):
    return self._contact_forces[:, self._feet_indices, 2] > 1.

  @property
  def foot_contact_forces(self):
    return self._contact_forces[:, self._feet_indices, :]

  @property
  def calf_contacts(self):
    return self._contact_forces[:, self._calf_indices, 2] > 1.

  @property
  def calf_contact_forces(self):
    return self._contact_forces[:, self._calf_indices, :]

  @property
  def thigh_contacts(self):
    return self._contact_forces[:, self._thigh_indices, 2] > 1.

  @property
  def thigh_contact_forces(self):
    return self._contact_forces[:, self._thigh_indices, :]

  @property
  def has_body_contact(self):
    return torch.any(torch.norm(self._contact_forces[:, self._body_indices, :],
                                dim=-1) > 1.,
                     dim=1)

  @property
  def hip_positions_in_body_frame(self):
    raise NotImplementedError()

  @property
  def all_foot_jacobian(self):
    rot_mat_t = self.base_rot_mat_t
    jacobian = torch.zeros((self._num_envs, 12, 12), device=self._device)
    jacobian[:, :3, :3] = torch.bmm(rot_mat_t, self._jacobian[:, 4, :3, 6:9])
    jacobian[:, 3:6, 3:6] = torch.bmm(rot_mat_t, self._jacobian[:, 8, :3,
                                                                9:12])
    jacobian[:, 6:9, 6:9] = torch.bmm(rot_mat_t, self._jacobian[:, 12, :3,
                                                                12:15])
    jacobian[:, 9:12, 9:12] = torch.bmm(rot_mat_t, self._jacobian[:, 16, :3,
                                                                  15:18])
    return jacobian

  @property
  def motor_group(self):
    return self._motors

  @property
  def num_envs(self):
    return self._num_envs

  @property
  def num_dof(self):
    return self._num_dof

  @property
  def device(self):
    return self._device

  @property
  def time_since_reset(self):
    return torch.clone(self._time_since_reset)

  @property
  def control_timestep(self):
    return self._sim_config.dt * self._sim_config.action_repeat
